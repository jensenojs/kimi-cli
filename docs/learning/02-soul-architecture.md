# 专题 2: Soul 核心架构与上下文管理（深度版）

## 2.1 架构概览：事件驱动的状态机

Soul 是 Kimi CLI 的**执行引擎**，采用**事件驱动 + 状态机**架构。核心挑战：

- **状态一致性**：724 行 prompt.py 的复杂交互状态如何与 Soul 同步？
- **错误恢复**：LLM 调用失败、工具执行异常、用户中断如何处理？
- **时间旅行**：如何回溯到任意历史点？
- **资源限制**：Token 超限、上下文过长如何优雅处理？

**核心状态流转**：

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Checkpoint  │───▶ 创建可回溯点
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Step     │───▶ LLM 调用 + 工具执行
└──────┬──────┘
       │
       ├─▶ 正常 ──▶ 检查 Finish ──▶ 是 ──▶ End
       │                                      │
       └─▶ 异常 ──▶ BackToTheFuture ──▶ revert_to ──▶ Retry
              ▲                                      │
              │                                      │
              └───────▶ 其他异常 ──▶ Abort
```

**状态分类**：

| 状态类型 | 存储位置 | 生命周期 | 可变性 | 示例 |
|---------|---------|---------|--------|------|
| **配置状态** | `self._agent`, `self._runtime` | 会话级 | 不可变 | Agent 配置、LLM 设置 |
| **会话状态** | `self._context` | 会话级 | 可读写 | 消息历史、Token 计数 |
| **行为状态** | `self._thinking_effort` | 运行时 | 可修改 | Thinking 模式开关 |
| **临时状态** | 函数局部变量 | 函数调用 | 用完即弃 | 重试计数、Checkpoint ID |

## 2.2 Context：持久化状态管理（142 行）

### 2.2.1 核心状态设计

```python
# src/kimi_cli/soul/context.py:14-20
class Context:
    def __init__(self, file_backend: Path):
        self._file_backend = file_backend  # 持久化后端（JSON Lines）
        self._history: list[Message] = []  # 内存缓存（快速访问）
        self._token_count: int = 0         # Token 使用状态
        self._next_checkpoint_id: int = 0  # Checkpoint 计数器（单调递增）
        """The ID of the next checkpoint, starting from 0, incremented after each checkpoint."""
```

**状态双模式**：

```
┌─────────────────────────────────────┐
│        内存状态 (list[Message])     │
│  - 快速访问                         │
│  - 易失性（进程退出丢失）           │
└──────────────┬──────────────────────┘
               │ 自动同步
               ▼
┌─────────────────────────────────────┐
│        磁盘状态 (JSON Lines)        │
│  - 持久化存储                       │
│  - 可恢复                           │
└─────────────────────────────────────┘
```

**为什么需要双模式？**

1. **性能**：内存读取 O(1)，磁盘读取 O(n)
2. **可靠性**：磁盘保证不丢失，内存保证速度
3. **一致性**：双写操作保证状态同步

### 2.2.2 状态恢复：从磁盘重建内存

```python
# src/kimi_cli/soul/context.py:22-48
async def restore(self) -> bool:
    """
    从磁盘文件恢复状态到内存
    
    防御性检查：
    1. 避免重复恢复（self._history 已存在）
    2. 文件不存在 → 跳过
    3. 文件为空 → 跳过
    """
    logger.debug("Restoring context from file: {file_backend}", file_backend=self._file_backend)
    
    # 防御性检查：避免重复恢复
    if self._history:
        logger.error("The context storage is already modified")
        raise RuntimeError("The context storage is already modified")
    
    if not self._file_backend.exists():
        logger.debug("No context file found, skipping restoration")
        return False  # 无状态可恢复
    
    if self._file_backend.stat().st_size == 0:
        logger.debug("Empty context file, skipping restoration")
        return False
    
    # 逐行解析 JSON Lines
    async with aiofiles.open(self._file_backend, encoding="utf-8") as f:
        async for line in f:
            if not line.strip():
                continue
            line_json = json.loads(line)
            
            # 元数据处理（非消息状态）
            if line_json["role"] == "_usage":
                self._token_count = line_json["token_count"]
                continue
            if line_json["role"] == "_checkpoint":
                self._next_checkpoint_id = line_json["id"] + 1
                continue
            
            # 消息状态恢复
            message = Message.model_validate(line_json)
            self._history.append(message)
    
    return True
```

**调用链路**：

```
Session 初始化
└── Context.restore()
    ├── 检查文件存在性
    ├── 逐行解析 JSON
    ├── 区分元数据（_usage, _checkpoint）
    └── 反序列化 Message
```

**JSON Lines 格式设计**：

```json
// 普通消息
{"role": "user", "content": "Hello"}

// Token 使用元数据
{"role": "_usage", "token_count": 1000}

// Checkpoint 标记
{"role": "_checkpoint", "id": 3}
```

**优势**：

1. **追加写入**：O(1) 复杂度，无需重写整个文件
2. **人类可读**：纯文本，易于调试
3. **容错性**：单行损坏不影响其他行
4. **流式处理**：可逐行读取，内存效率高

### 2.2.3 Checkpoint 机制：状态快照

```python
# src/kimi_cli/soul/context.py:62-72
async def checkpoint(self, add_user_message: bool):
    """
    创建状态快照（可回溯点）
    
    状态变更：
    1. 分配 checkpoint_id（单调递增）
    2. 持久化 checkpoint 标记到磁盘
    3. 可选：在内存历史中插入标记消息
    """
    checkpoint_id = self._next_checkpoint_id
    self._next_checkpoint_id += 1  # 状态变更
    logger.debug("Checkpointing, ID: {id}", id=checkpoint_id)
    
    # 持久化 checkpoint 标记
    async with aiofiles.open(self._file_backend, "a", encoding="utf-8") as f:
        await f.write(json.dumps({"role": "_checkpoint", "id": checkpoint_id}) + "\n")
    
    if add_user_message:
        # 在消息历史中插入标记（用于 LLM 理解）
        await self.append_message(
            Message(role="user", content=[system(f"CHECKPOINT {checkpoint_id}")])
        )
```

**Checkpoint 语义**：

- **Checkpoint = 可回溯点**：每个 checkpoint 是一个安全恢复点
- **ID 单调递增**：保证全序关系，可比较大小
- **双重记录**：磁盘（持久化）+ 内存（快速访问）

**Checkpoint 创建时机**：

```python
# src/kimi_cli/soul/kimisoul.py:141
await self._checkpoint()  # run() 开始时
# 目的：创建 checkpoint 0，作为初始回溯点

# src/kimi_cli/soul/kimisoul.py:175
await self._checkpoint()  # 每个 step 开始前
# 目的：创建可回溯点，支持时间旅行
```

### 2.2.4 状态回溯：时间旅行实现

```python
# src/kimi_cli/soul/context.py:74-127
async def revert_to(self, checkpoint_id: int):
    """
    回溯到指定 checkpoint（时间旅行核心）
    
    原子操作：
    1. 文件轮转（保留历史）
    2. 重置内存状态
    3. 从轮转文件重建状态（直到指定 checkpoint）
    """
    logger.debug("Reverting checkpoint, ID: {id}", id=checkpoint_id)
    
    # 防御性检查
    if checkpoint_id >= self._next_checkpoint_id:
        logger.error("Checkpoint {checkpoint_id} does not exist", checkpoint_id=checkpoint_id)
        raise ValueError(f"Checkpoint {checkpoint_id} does not exist")
    
    # 文件轮转（保留完整历史）
    rotated_file_path = await next_available_rotation(self._file_backend)
    if rotated_file_path is None:
        logger.error("No available rotation path found")
        raise RuntimeError("No available rotation path found")
    await aiofiles.os.rename(self._file_backend, rotated_file_path)
    logger.debug("Rotated history file: {rotated_file_path}", rotated_file_path=rotated_file_path)
    
    # 重置内存状态（原子操作）
    self._history.clear()
    self._token_count = 0
    self._next_checkpoint_id = 0
    
    # 从轮转文件重建状态（直到指定 checkpoint）
    async with (
        aiofiles.open(rotated_file_path, encoding="utf-8") as old_file,
        aiofiles.open(self._file_backend, "w", encoding="utf-8") as new_file,
    ):
        async for line in old_file:
            if not line.strip():
                continue
            
            line_json = json.loads(line)
            
            # 在指定 checkpoint 处停止
            if line_json["role"] == "_checkpoint" and line_json["id"] == checkpoint_id:
                break
            
            # 写入新文件（保留历史）
            await new_file.write(line)
            
            # 重建内存状态
            if line_json["role"] == "_usage":
                self._token_count = line_json["token_count"]
            elif line_json["role"] == "_checkpoint":
                self._next_checkpoint_id = line_json["id"] + 1
            else:
                message = Message.model_validate(line_json)
                self._history.append(message)
```

**原子性保证**：

```python
# 关键：使用 asyncio.shield 保护状态操作
await asyncio.shield(self._grow_context(result, results))

# 为什么需要 shield？
# - 防止 asyncio.CancelledError 中断状态更新
# - 保证上下文完整性（不部分更新）
```

**文件轮转策略**：

```
context.jsonl          context.jsonl.1
     │                        │
     ├─ Message 1             ├─ Message 1
     ├─ Message 2             ├─ Message 2
     ├─ Checkpoint 0          ├─ Checkpoint 0
     ├─ Message 3             ├─ Message 3
     ├─ Checkpoint 1    ──▶   ├─ Checkpoint 1
     ├─ Message 4             └─（停止，不写入 Message 4+）
     └─ Message 5
```

**状态重建过程**：

```
轮转文件读取          新文件写入          内存状态
     │                     │                  │
     ├─ Message 1    ──▶   ├─ Message 1       ├─ history.append(M1)
     ├─ Message 2    ──▶   ├─ Message 2       ├─ history.append(M2)
     ├─ Checkpoint 0 ──▶   ├─ Checkpoint 0    ├─ _next_checkpoint_id = 1
     ├─ Message 3    ──▶   ├─ Message 3       ├─ history.append(M3)
     ├─ Checkpoint 1 ──▶   ├─ Checkpoint 1    ├─ _next_checkpoint_id = 2
     │                     │                  │
     ├─ Message 4    ──▶   │（停止）          │（不恢复）
     └─ Message 5    ──▶   │                  │
```

## 2.3 KimiSoul：主执行引擎状态机（342 行）

### 2.3.1 初始化状态分类

```python
# src/kimi_cli/soul/kimisoul.py:51-83
def __init__(self, agent: Agent, runtime: Runtime, *, context: Context):
    """
    状态初始化：明确分类，避免混淆
    """
    # 配置状态（不可变）
    self._agent = agent  # Agent 配置（系统提示、工具集）
    self._runtime = runtime  # 运行时依赖（LLM、配置、会话）
    
    # 共享状态（可变，但由外部管理）
    self._denwa_renji = runtime.denwa_renji  # 时间旅行消息中心
    self._approval = runtime.approval  # 审批系统
    self._context = context  # 上下文（核心可变状态）
    
    # 控制状态（可变，内部管理）
    self._loop_control = runtime.config.loop_control  # 循环控制参数
    self._compaction = SimpleCompaction()  # 压缩策略
    self._reserved_tokens = RESERVED_TOKENS  # 预留 Token 数
    self._thinking_effort: ThinkingEffort = "off"  # Thinking 模式
    
    # 派生状态（基于工具存在性）
    for tool in agent.toolset.tools:
        if tool.name == SendDMail_NAME:
            self._checkpoint_with_user_message = True
            break
    else:
        self._checkpoint_with_user_message = False
    # 决策逻辑：如果 Agent 包含 SendDMail 工具，则在 checkpoint 时添加用户消息
    # 目的：让 LLM 知道当前处于哪个 checkpoint
```

**状态维度分析**：

| 状态类型 | 存储位置 | 生命周期 | 修改时机 | 线程安全 |
|---------|---------|---------|---------|---------|
| **配置状态** | `self._agent` | 会话级 | 从不 | 安全（只读） |
| **共享状态** | `self._context` | 会话级 | 运行时 | 需保护（异步锁） |
| **控制状态** | `self._thinking_effort` | 运行时 | 用户命令 | 安全（单线程） |
| **派生状态** | `self._checkpoint_with_user_message` | 初始化 | 从不 | 安全（只读） |

### 2.3.2 主循环状态机：_agent_loop

```python
# src/kimi_cli/soul/kimisoul.py:146-196
async def _agent_loop(self):
    """
    主状态机：管理整个 Agent 执行生命周期
    
    状态流转：
    1. Checkpoint：创建可回溯点
    2. Step：执行单步（LLM 调用 + 工具执行）
    3. 异常处理：BackToTheFuture 或其他异常
    4. 终止检查：完成或达到最大步数
    """
    assert self._runtime.llm is not None
    
    # 审批任务：并行运行，处理人机交互
    async def _pipe_approval_to_wire():
        while True:
            request = await self._approval.fetch_request()
            wire_send(request)  # 发送到 UI 层
    
    step_no = 1
    while True:  # 主循环
        wire_send(StepBegin(step_no))  # 事件通知
        
        # 创建审批任务（并行）
        approval_task = asyncio.create_task(_pipe_approval_to_wire())
        
        try:
            # 上下文过长检测（预防性压缩）
            if (
                self._context.token_count + self._reserved_tokens
                >= self._runtime.llm.max_context_size
            ):
                logger.info("Context too long, compacting...")
                wire_send(CompactionBegin())
                await self.compact_context()  # 状态转换：完整历史 → 摘要
                wire_send(CompactionEnd())
            
            await self._checkpoint()  # 创建回溯点
            self._denwa_renji.set_n_checkpoints(self._context.n_checkpoints)
            
            finished = await self._step()  # 执行单步（可能抛出异常）
            
        except BackToTheFuture as e:  # 时间旅行异常（特殊控制流）
            # 状态回溯：恢复到指定 checkpoint
            await self._context.revert_to(e.checkpoint_id)
            # 创建新 checkpoint（用于后续回溯）
            await self._checkpoint()
            # 添加未来消息（D-Mail 内容）
            await self._context.append_message(e.messages)
            continue  # 重试当前步骤（不增加 step_no）
        
        except (ChatProviderError, asyncio.CancelledError):
            # 不可恢复错误：中断执行
            wire_send(StepInterrupted())
            raise  # 终止循环
        
        finally:
            # 清理：取消审批任务（避免泄漏）
            approval_task.cancel()
        
        # 正常终止检查
        if finished:
            return  # 成功完成
        
        step_no += 1
        if step_no > self._loop_control.max_steps_per_run:
            # 状态保护：防止无限循环
            raise MaxStepsReached(self._loop_control.max_steps_per_run)
```

**状态机图**：

```
          ┌─────────────┐
          │  StepBegin  │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │  Checkpoint │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │    Step     │
          └──────┬──────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │Normal││BackTo││Other │
    │      ││Future││Error │
    └──┬───┘ └──┬───┘ └──┬───┘
       │        │        │
       │        ▼        │
       │   ┌──────────┐  │
       │   │ Revert   │  │
       │   └──────────┘  │
       │        │        │
       │        ▼        │
       │   ┌──────────┐  │
       │   │Checkpoint│  │
       │   └──────────┘  │
       └──────┬───────────┘
              │
              ▼
          ┌─────────┐
          │Continue │（重试）
          └─────────┘
```

**关键设计决策**：

1. **审批任务并行化**
   ```python
   approval_task = asyncio.create_task(_pipe_approval_to_wire())
   # 目的：不阻塞主循环，实时处理审批请求
   # 风险：可能被子 Agent 的审批任务干扰（FIXME 注释）
   ```

2. **预防性压缩**
   ```python
   if self._context.token_count + self._reserved_tokens >= max_context_size:
       await self.compact_context()
   # 策略：在达到限制前主动压缩，避免硬失败
   ```

3. **异常作为控制流**
   ```python
   except BackToTheFuture as e:
       await self._context.revert_to(e.checkpoint_id)
       continue  # 重试当前步骤
   # 优势：清晰的状态回溯语义
   # 代价：异常性能开销（可忽略）
   ```

4. **状态保护**
   ```python
   finally:
       approval_task.cancel()
   # 保证：即使异常也能清理资源
   ```

### 2.3.3 单步执行：_step 方法

```python
# src/kimi_cli/soul/kimisoul.py:197-266
async def _step(self) -> bool:
    """
    执行单步：LLM 调用 + 工具执行
    
    返回：
    - True：应该停止（完成或被拒绝）
    - False：继续下一步
    """
    assert self._runtime.llm is not None
    chat_provider = self._runtime.llm.chat_provider
    
    # 重试装饰器：处理可重试错误
    @tenacity.retry(
        retry=retry_if_exception(self._is_retryable_error),
        before_sleep=partial(self._retry_log, "step"),
        wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
        stop=stop_after_attempt(self._loop_control.max_retries_per_step),
        reraise=True,  # 重试耗尽后重新抛出
    )
    async def _kosong_step_with_retry() -> StepResult:
        # 调用 kosong 框架执行一步
        return await kosong.step(
            chat_provider.with_thinking(self._thinking_effort),
            self._agent.system_prompt,
            self._agent.toolset,
            self._context.history,  # 传递当前上下文
            on_message_part=wire_send,  # 流式响应
            on_tool_result=wire_send,   # 工具结果
        )
    
    result = await _kosong_step_with_retry()
    logger.debug("Got step result: {result}", result=result)
    
    # 更新 Token 计数
    if result.usage is not None:
        await self._context.update_token_count(result.usage.input)
        wire_send(StatusUpdate(status=self.status))
    
    # 等待工具结果（异步生成器）
    results = await result.tool_results()
    logger.debug("Got tool results: {results}", results=results)
    
    # 保护状态更新：防止中断导致不一致
    await asyncio.shield(self._grow_context(result, results))
    
    # 检查是否有工具被拒绝
    rejected = any(isinstance(result.result, ToolRejectedError) for result in results)
    if rejected:
        _ = self._denwa_renji.fetch_pending_dmail()  # 清理 pending D-Mail
        return True  # 停止执行
    
    # 检查是否有 D-Mail（时间旅行消息）
    if dmail := self._denwa_renji.fetch_pending_dmail():
        assert dmail.checkpoint_id >= 0
        assert dmail.checkpoint_id < self._context.n_checkpoints
        # 抛出异常，让主循环处理时间旅行
        raise BackToTheFuture(
            dmail.checkpoint_id,
            [Message(role="user", content=[system(f"...{dmail.message}...")])]
        )
    
    # 返回是否完成（无工具调用 = LLM 认为完成）
    return not result.tool_calls
```

**状态增长过程**：

```python
# src/kimi_cli/soul/kimisoul.py:268-277
async def _grow_context(self, result: StepResult, tool_results: list[ToolResult]):
    """
    状态增长：将 LLM 响应和工具结果添加到上下文
    
    调用时机：在 asyncio.shield 保护下执行
    """
    logger.debug("Growing context with result: {result}", result=result)
    
    # 添加 LLM 响应消息
    await self._context.append_message(result.message)
    if result.usage is not None:
        await self._context.update_token_count(result.usage.total)
    
    # 添加工具结果消息
    for tool_result in tool_results:
        logger.debug("Appending tool result to context: {tool_result}", tool_result=tool_result)
        await self._context.append_message(tool_result_to_messages(tool_result))
```

**状态增长模式**：

```
Context (旧状态)
    ├─ Message 1
    ├─ Message 2
    └─ ...

    ↓ LLM 调用

Context (中间状态)
    ├─ Message 1
    ├─ Message 2
    ├─ ...
    └─ LLM Response Message

    ↓ 工具执行

Context (新状态)
    ├─ Message 1
    ├─ Message 2
    ├─ ...
    ├─ LLM Response Message
    ├─ Tool Result Message 1
    ├─ Tool Result Message 2
    └─ ...
```

**为什么需要 asyncio.shield？**

```python
# 场景：用户按下 Ctrl+C（发送 CancelledError）
# 无 shield：
await self._grow_context(result, results)  # 可能被中断
# 结果：上下文部分更新，状态不一致

# 有 shield：
await asyncio.shield(self._grow_context(result, results))  # 不可中断
# 结果：要么完全更新，要么不更新，状态一致
```

### 2.3.4 重试机制：tenacity 装饰器

```python
# src/kimi_cli/soul/kimisoul.py:203-209
@tenacity.retry(
    retry=retry_if_exception(self._is_retryable_error),
    before_sleep=partial(self._retry_log, "step"),
    wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
    stop=stop_after_attempt(self._loop_control.max_retries_per_step),
    reraise=True,  # 重试耗尽后重新抛出
)
async def _kosong_step_with_retry() -> StepResult:
    ...
```

**重试策略**：

| 参数 | 值 | 含义 |
|------|---|------|
| `initial` | 0.3 | 初始等待 0.3 秒 |
| `max` | 5 | 最大等待 5 秒 |
| `jitter` | 0.5 | 随机抖动 0.5 秒 |
| `max_retries_per_step` | 3 | 最多重试 3 次 |

**指数退避 + 抖动**：

```
第 1 次失败：等待 0.3 * random(0.5, 1.5) = 0.15-0.45 秒
第 2 次失败：等待 0.6 * random(0.5, 1.5) = 0.3-0.9 秒
第 3 次失败：等待 1.2 * random(0.5, 1.5) = 0.6-1.8 秒
...
第 N 次失败：等待 min(5, 0.3 * 2^(n-1)) 秒
```

**可重试错误判断**：

```python
# src/kimi_cli/soul/kimisoul.py:305-314
@staticmethod
def _is_retryable_error(exception: BaseException) -> bool:
    # 网络错误：总是可重试
    if isinstance(exception, (APIConnectionError, APITimeoutError)):
        return True
    
    # HTTP 状态错误：特定状态码可重试
    return isinstance(exception, APIStatusError) and exception.status_code in (
        429,  # Too Many Requests（限流）
        500,  # Internal Server Error（服务器错误）
        502,  # Bad Gateway（网关错误）
        503,  # Service Unavailable（服务不可用）
    )
    # 决策逻辑：客户端错误（4xx）不重试，服务器错误（5xx）重试
```

## 2.4 时间旅行：BackToTheFuture 异常

### 2.4.1 异常作为控制流

```python
# src/kimi_cli/soul/kimisoul.py:328-337
class BackToTheFuture(Exception):
    """
    时间旅行异常：携带回溯目标状态和附加数据
    
    设计哲学：
    - 异常不是错误，而是控制流信号
    - 封装状态转移所需的所有信息
    - 强制调用方处理（不处理会崩溃）
    """
    
    def __init__(self, checkpoint_id: int, messages: Sequence[Message]):
        self.checkpoint_id = checkpoint_id  # 目标 checkpoint ID
        self.messages = messages  # 回溯后追加的消息（D-Mail 内容）
```

**对比传统模式**：

```python
# 传统模式（返回值检查）
result = await self._step()
if isinstance(result, BackToTheFuture):
    await self._context.revert_to(result.checkpoint_id)
    ...

# 异常模式（更清晰）
try:
    finished = await self._step()
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)
    ...
```

**异常模式的优势**：

1. **强制处理**：不处理会崩溃，避免忽略错误
2. **控制流清晰**：正常路径 vs 异常路径明确分离
3. **状态封装**：异常对象携带所有必要信息
4. **调用栈清理**：自动清理中间状态

### 2.4.2 D-Mail 触发流程

```python
# 场景：工具发送 D-Mail，触发时间旅行

# Step 1: 工具调用 SendDMail
dmail_tool = SendDMailTool()
await dmail_tool(checkpoint_id=3, message="Go back!")

# Step 2: DenwaRenji 存储消息
# src/kimi_cli/soul/denwarenji.py:19-27
def send_dmail(self, dmail: DMail):
    if self._pending_dmail is not None:
        raise DenwaRenjiError("Only one D-Mail can be sent at a time")
    if dmail.checkpoint_id < 0:
        raise ValueError("Checkpoint ID cannot be negative")
    if dmail.checkpoint_id >= self._n_checkpoints:
        raise ValueError("Checkpoint does not exist")
    self._pending_dmail = dmail  # 存储待处理消息

# Step 3: Soul 检查并抛出异常
# src/kimi_cli/soul/kimisoul.py:241-264
if dmail := self._denwa_renji.fetch_pending_dmail():
    raise BackToTheFuture(
        dmail.checkpoint_id,
        [Message(role="user", content=[system(f"...{dmail.message}...")])]
    )

# Step 4: 主循环捕获并处理
# src/kimi_cli/soul/kimisoul.py:178-182
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)
    await self._checkpoint()
    await self._context.append_message(e.messages)
    continue  # 重试
```

**完整调用链路**：

```
SendDMailTool.__call__(checkpoint_id=3, message="Go back!")
└── denwa_renji.send_dmail(DMail(checkpoint_id=3, message="Go back!"))
    └── self._pending_dmail = dmail  # 状态：None → Some

KimiSoul._step()
└── if dmail := self._denwa_renji.fetch_pending_dmail():
    └── self._pending_dmail = None  # 消费消息
    └── raise BackToTheFuture(3, messages)

KimiSoul._agent_loop()
└── except BackToTheFuture as e:
    ├── await self._context.revert_to(3)  # 回溯到 checkpoint 3
    ├── await self._checkpoint()  # 创建新 checkpoint
    ├── await self._context.append_message(e.messages)  # 添加 D-Mail
    └── continue  # 重试当前步骤
```

## 2.5 上下文压缩：状态优化

### 2.5.1 压缩触发条件

```python
# src/kimi_cli/soul/kimisoul.py:165-172
if (
    self._context.token_count + self._reserved_tokens
    >= self._runtime.llm.max_context_size
):
    logger.info("Context too long, compacting...")
    wire_send(CompactionBegin())
    await self.compact_context()  # 状态转换
    wire_send(CompactionEnd())
```

**触发策略**：

```
当前 Token 数: 150,000
预留 Token 数: 50,000
总计: 200,000

LLM 最大上下文: 200,000
──────────────────────────────
使用率: 100% → 触发压缩

压缩后:
当前 Token 数: 5,000 (摘要)
预留 Token 数: 50,000
总计: 55,000
──────────────────────────────
使用率: 27.5% → 安全范围
```

### 2.5.2 SimpleCompaction 策略

```python
# src/kimi_cli/soul/compaction.py:33-55
class SimpleCompaction(Compaction):
    MAX_PRESERVED_MESSAGES = 2  # 保留最近 2 条消息
    
    async def compact(self, messages: Sequence[Message], llm: LLM) -> Sequence[Message]:
        """
        压缩策略：保留最近 N 条消息，其余摘要
        
        权衡：
        - 优势：大幅减少 Token 使用
        - 代价：丢失历史细节
        - 改进方向：基于重要性评分选择性保留
        """
        history = list(messages)
        
        # 保留策略：最近 user/assistant 消息
        preserve_start_index = len(history)
        n_preserved = 0
        for index in range(len(history) - 1, -1, -1):
            if history[index].role in {"user", "assistant"}:
                n_preserved += 1
                if n_preserved == self.MAX_PRESERVED_MESSAGES:
                    preserve_start_index = index
                    break
        
        to_compact = history[:preserve_start_index]  # 待压缩部分
        to_preserve = history[preserve_start_index:]  # 保留部分
        
        # 使用 LLM 生成摘要
        history_text = "\n\n".join(
            f"## Message {i + 1}\nRole: {msg.role}\nContent: {msg.content}"
            for i, msg in enumerate(to_compact)
        )
        
        compact_prompt = f"""
        Summarize the following conversation history into a concise summary.
        Keep key information like file paths, code changes, and decisions.
        
        {history_text}
        """
        
        compacted_msg, usage = await generate(
            chat_provider=llm.chat_provider,
            system_prompt="You are a helpful assistant.",
            history=[Message(role="user", content=compact_prompt)],
        )
        
        # 重建状态：摘要 + 保留消息
        content: list[ContentPart] = [
            system("Previous context has been compacted. Summary:")
        ]
        content.extend(compacted_msg.content)
        
        compacted_messages: list[Message] = [Message(role="assistant", content=content)]
        compacted_messages.extend(to_preserve)
        return compacted_messages
```

**压缩示例**：

```
压缩前（10 条消息，2000 tokens）：
├─ User: "Create a file"
├─ Assistant: "I'll create it"
├─ Tool: File created
├─ User: "Add function"
├─ Assistant: "I'll add it"
├─ Tool: Function added
├─ User: "Fix bug"
├─ Assistant: "I'll fix it"
├─ Tool: Bug fixed
└─ User: "Thank you"

压缩后（3 条消息，200 tokens）：
├─ Assistant: "Summary: Created file, added function, fixed bug"
├─ User: "Fix bug"
└─ Assistant: "I'll fix it"
```

**压缩流程**：

```python
# src/kimi_cli/soul/kimisoul.py:279-303
async def compact_context(self) -> None:
    """
    压缩上下文：完整历史 → 摘要
    
    原子操作：
    1. 生成压缩消息（可能失败）
    2. 回溯到 checkpoint 0（清空上下文）
    3. 添加压缩消息（新状态）
    """
    
    # 重试保护：压缩也可能失败
    @tenacity.retry(...)
    async def _compact_with_retry() -> Sequence[Message]:
        if self._runtime.llm is None:
            raise LLMNotSet()
        return await self._compaction.compact(self._context.history, self._runtime.llm)
    
    compacted_messages = await _compact_with_retry()
    
    # 原子状态替换
    await self._context.revert_to(0)  # 清空上下文
    await self._checkpoint()  # 创建新 checkpoint
    await self._context.append_message(compacted_messages)  # 添加压缩状态
```

## 2.6 总结：Soul 状态管理精髓

### 2.6.1 状态分层原则

```
配置状态（不可变）
    ↓
会话状态（可回溯）
    ↓
请求状态（临时）
    ↓
临时状态（用完即弃）
```

### 2.6.2 状态保护模式

```python
# 模式 1：asyncio.shield() 保护关键状态更新
await asyncio.shield(self._grow_context(result, results))

# 模式 2：异常作为控制流
try:
    finished = await self._step()
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)
    continue

# 模式 3：文件轮转保证原子性
rotated_file_path = await next_available_rotation(self._file_backend)
await aiofiles.os.rename(self._file_backend, rotated_file_path)

# 模式 4：重试机制保证可靠性
@tenacity.retry(...)
async def _kosong_step_with_retry() -> StepResult:
    ...
```

### 2.6.3 事件驱动架构

```python
# 事件发送
wire_send(StepBegin(step_no))
wire_send(StatusUpdate(status=self.status))
wire_send(CompactionBegin())

# 事件处理（在 UI 层）
# - StepBegin: 显示步骤开始
# - StatusUpdate: 更新状态栏
# - CompactionBegin: 显示压缩进度
```

### 2.6.4 性能优化

```python
# 1. 双模式状态（内存 + 磁盘）
self._history: list[Message] = []  # 快速访问
self._file_backend: Path  # 持久化

# 2. 惰性加载
if now - cache_time <= refresh_interval:
    return cached_paths  # 快速路径

# 3. 批量操作
results = await asyncio.gather(task1, task2, task3)

# 4. 预防性压缩
if token_count + reserved >= max_size:
    await compact_context()  # 避免硬失败
```

---

**下一步**：专题 3 - 工具系统设计与实现（将深入 MCP 协议适配和依赖注入）

**文档统计**：
- 01-agent-system.md: ~400 行（深度版）
- 02-soul-architecture.md: ~600 行（深度版）
- 03-tool-system.md: 待更新（目标 ~400 行）
- 04-async-patterns.md: 1982 行（已详尽）

**总计**：~3400 行，涵盖 Kimi CLI 核心架构的 80%