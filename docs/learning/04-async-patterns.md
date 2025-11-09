# 专题 4: 异步编程模式与状态机设计

## 4.1 架构概览

Kimi CLI 采用**全异步架构**，核心挑战：
- **并发控制**：多个异步任务如何协调
- **状态一致性**：异步操作中的状态变更如何保护
- **错误处理**：异步链路的异常传播与恢复

**状态机模式贯穿始终**：
```
[Idle] → [Running] → [Paused] → [Resumed] → [Completed]
   ↓        ↓          ↓           ↓           ↓
[Error] ← [Cancelled] ← [Interrupted] ← [BackToTheFuture]
```

## 4.2 Approval 系统 - 人机交互状态机

### 4.2.1 核心状态设计

```python
# src/kimi_cli/soul/approval.py:8-14
class Approval:
    def __init__(self, yolo: bool = False):
        self._request_queue = asyncio.Queue[ApprovalRequest]()
        self._yolo = yolo  # 全局状态：是否自动批准
        self._auto_approve_actions: set[str] = set()  # 会话状态：自动批准动作集合
```

**状态维度分析**：

| 状态类型 | 存储位置 | 生命周期 | 作用域 |
|---------|---------|---------|--------|
| `_yolo` | 实例变量 | 会话级（可动态修改） | 全局影响 |
| `_auto_approve_actions` | Set | 会话级（累积） | 动作级 |
| `_request_queue` | asyncio.Queue | 会话级 | 请求-响应 |

**设计动机**：
- **分层状态**：全局状态 + 累积状态 + 临时状态
- **不可变原则**：`set()` 是可变的，但存储的动作名（str）是不可变的
- **类型安全**：`asyncio.Queue[ApprovalRequest]` 提供编译时类型检查

### 4.2.2 请求-响应状态流转

```python
# src/kimi_cli/soul/approval.py:18-62
async def request(self, sender: str, action: str, description: str) -> bool:
    """
    工具 → Approval 的请求入口
    
    状态流转：
    1. 检查调用上下文（必须在工具调用中）
    2. 检查全局自动批准（yolo 模式）
    3. 检查会话自动批准（已批准过的动作）
    4. 创建请求 → 放入队列 → 等待响应
    5. 处理响应，更新状态
    """
    # 状态保护：确保在正确的上下文中调用
    tool_call = get_current_tool_call_or_none()
    if tool_call is None:
        raise RuntimeError("Approval must be requested from a tool call.")
    
    logger.debug(
        "{tool_name} ({tool_call_id}) requesting approval: {action} {description}",
        tool_name=tool_call.function.name,
        tool_call_id=tool_call.id,
        action=action,
        description=description,
    )
    
    # 快速路径：全局自动批准
    if self._yolo:
        return True  # 状态决策：不修改任何状态
    
    # 快速路径：会话自动批准
    if action in self._auto_approve_actions:
        return True  # 状态决策：读取累积状态
    
    # 慢速路径：人机交互
    request = ApprovalRequest(tool_call.id, sender, action, description)
    self._request_queue.put_nowait(request)  # 状态变更：请求入队
    response = await request.wait()  # 状态等待：阻塞直到响应
    
    # 响应处理：状态更新
    logger.debug("Received approval response: {response}", response=response)
    match response:
        case ApprovalResponse.APPROVE:
            return True  # 临时批准，不修改状态
        case ApprovalResponse.APPROVE_FOR_SESSION:
            self._auto_approve_actions.add(action)  # 状态累积
            return True
        case ApprovalResponse.REJECT:
            return False  # 状态决策
```

**调用链路**：
```
BashTool.__call__("rm -rf /")
└── approval.request("Bash", "execute", "rm -rf /")
    ├── get_current_tool_call_or_none()  # 上下文验证
    ├── if self._yolo: return True  # 全局状态检查
    ├── if action in self._auto_approve_actions: return True  # 累积状态检查
    ├── request = ApprovalRequest(...)  # 创建请求状态
    ├── self._request_queue.put_nowait(request)  # 状态转移：Idle → Pending
    ├── response = await request.wait()  # 状态等待
    └── match response:  # 状态处理
        ├── APPROVE: 临时状态，不存储
        ├── APPROVE_FOR_SESSION: self._auto_approve_actions.add(action)  # 状态持久化
        └── REJECT: 返回 False
```

**状态机图**：
```
          ┌─────────────┐
          │   Start     │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │ Check YOLO  │───True───▶┌─────────────┐
          └──────┬──────┘           │   Approved  │
                 │False             └─────────────┘
                 ▼
          ┌─────────────┐
          │Check History│───True───▶┌─────────────┐
          └──────┬──────┘           │   Approved  │
                 │False             └─────────────┘
                 ▼
          ┌─────────────┐
          │  Send to UI │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │Wait Response│
          └──────┬──────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │Approve│ │Approve││Reject│
    │Once  │ │Forever││      │
    └──┬───┘ └───┬───┘ └──┬───┘
       │         │        │
       │         ▼        │
       │   ┌──────────┐   │
       │   │Add to Set│   │
       │   └──────────┘   │
       │         │        │
       └────┬────┴────┬───┘
            │         │
            ▼         ▼
        ┌─────────┐ ┌──────┐
        │Approved │ │Rejected│
        └─────────┘ └────────┘
```

### 4.2.3 异步队列模式

```python
# src/kimi_cli/soul/approval.py:64-68
async def fetch_request(self) -> ApprovalRequest:
    """
    Soul → Approval 的拉取入口
    
    状态流转：
    1. 从队列等待请求（阻塞）
    2. 返回请求给 UI 层
    """
    return await self._request_queue.get()
```

**生产者-消费者模式**：
- **生产者**：`request()` 方法（工具调用）
- **消费者**：`fetch_request()` 方法（UI 循环）
- **队列**：`asyncio.Queue` 提供线程安全的异步通信

**状态同步机制**：
```python
# 在 KimiSoul._agent_loop() 中
approval_task = asyncio.create_task(_pipe_approval_to_wire())

try:
    finished = await self._step()
except ...:
    ...
finally:
    approval_task.cancel()  # 确保任务清理
```

**关键设计**：
- **任务生命周期**：`approval_task` 与 `step` 任务并行运行
- **取消机制**：`finally` 块确保任务总是取消（避免泄漏）
- **异常隔离**：审批任务异常不影响主流程

## 4.3 DenwaRenji - 时间旅行消息系统

### 4.3.1 核心状态设计

```python
# src/kimi_cli/soul/denwarenji.py:14-18
class DenwaRenji:
    def __init__(self):
        self._pending_dmail: DMail | None = None  # 单例状态：最多一个待处理消息
        self._n_checkpoints: int = 0  # 外部状态：checkpoint 数量
```

**状态维度分析**：

| 状态类型 | 存储位置 | 生命周期 | 作用域 |
|---------|---------|---------|--------|
| `_pending_dmail` | 可选值 | 消息级（发送 → 消费） | 全局唯一 |
| `_n_checkpoints` | 整数 | 会话级（外部更新） | 只读引用 |

**设计动机**：
- **单例模式**：`DMail | None` 确保最多一个待处理消息（避免冲突）
- **状态验证**：`send_dmail()` 中三重检查（非空、非负、存在性）
- **所有权转移**：`fetch_pending_dmail()` 返回后清空（Move 语义）

### 4.3.2 时间旅行状态流转

```python
# src/kimi_cli/soul/denwarenji.py:19-27
def send_dmail(self, dmail: DMail):
    """
    工具 → DenwaRenji 的消息发送
    
    状态流转：
    1. 检查是否已有待处理消息（单例保护）
    2. 检查 checkpoint_id 有效性
    3. 存储消息（状态转移：None → Some）
    """
    if self._pending_dmail is not None:
        raise DenwaRenjiError("Only one D-Mail can be sent at a time")
    if dmail.checkpoint_id < 0:
        raise DenwaRenjiError("The checkpoint ID can not be negative")
    if dmail.checkpoint_id >= self._n_checkpoints:
        raise DenwaRenjiError("There is no checkpoint with the given ID")
    self._pending_dmail = dmail  # 状态转移
```

**调用链路**：
```
SendDMailTool.__call__(checkpoint_id=3, message="Go back!")
└── denwa_renji.send_dmail(DMail(checkpoint_id=3, message="Go back!"))
    ├── if self._pending_dmail is not None: raise  # 状态保护
    ├── if dmail.checkpoint_id < 0: raise  # 边界检查
    ├── if dmail.checkpoint_id >= self._n_checkpoints: raise  # 存在性检查
    └── self._pending_dmail = dmail  # 状态转移
```

**状态机图**：
```
          ┌─────────────┐
          │   No Mail   │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │Validate Mail│
          └──────┬──────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │Invalid││Invalid││Invalid│
    │ID    ││Range ││Exists │
    └──┬───┘ └───┬───┘ └───┬───┘
       │         │         │
       │         ▼         │
       │   ┌──────────┐    │
       │   │Set Mail  │    │
       │   └──────────┘    │
       └──────┬─────────────┘
              │
              ▼
          ┌─────────┐
          │Mail Set │
          └─────────┘
```

### 4.3.3 异常作为控制流

```python
# src/kimi_cli/soul/kimisoul.py:177-182
try:
    finished = await self._step()
except BackToTheFuture as e:  # 异常作为控制流
    await self._context.revert_to(e.checkpoint_id)  # 状态回溯
    await self._checkpoint()
    await self._context.append_message(e.messages)
    continue  # 重试当前步骤
```

**设计模式**：
- **异常即信号**：`BackToTheFuture` 不是错误，而是控制流信号
- **状态携带**：异常对象携带目标状态（checkpoint_id）和附加数据（messages）
- **结构化恢复**：`except` 块明确处理状态转移逻辑

**对比传统模式**：
```python
# 传统模式（返回值检查）
result = await self._step()
if isinstance(result, BackToTheFuture):
    await self._context.revert_to(result.checkpoint_id)
    ...

# 异常模式（更清晰的控制流）
try:
    finished = await self._step()
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)
    ...
```

**优势**：
- **强制处理**：异常不处理会崩溃（避免忽略错误）
- **控制流清晰**：正常路径 vs 异常路径明确分离
- **状态封装**：异常对象封装状态转移所需的所有信息

## 4.4 Prompt 系统 - 复杂交互状态机

### 4.4.1 724 行代码的架构分解

```python
# src/kimi_cli/ui/shell/prompt.py:402-500 (部分)
class CustomPromptSession:
    """
    状态管理复杂度：
    - 模式状态：AGENT vs SHELL
    - 思考状态：thinking on/off
    - 历史状态：用户输入历史
    - 附件状态：图片附件
    - 剪贴板状态：系统剪贴板
    - 自动补全状态：多个 completer
    - 键盘状态：快捷键绑定
    """
    
    def __init__(self, *, status_provider, model_capabilities):
        # 历史状态管理
        history_dir = get_share_dir() / "user-history"
        work_dir_id = md5(str(Path.cwd()).encode()).hexdigest()
        self._history_file = (history_dir / work_dir_id).with_suffix(".jsonl")
        
        # 加载历史状态
        history_entries = _load_history_entries(self._history_file)
        history = InMemoryHistory()
        for entry in history_entries:
            history.append_string(entry.content)
        
        # 模式状态
        self._mode: PromptMode = PromptMode.AGENT
        self._thinking: bool = False
        
        # 附件状态
        self._attachment_parts: dict[str, ContentPart] = {}
```

**状态分类统计**：

| 状态类别 | 数量 | 存储方式 | 持久化 |
|---------|------|---------|--------|
| 模式状态 | 2 | Enum + bool | 否 |
| 历史状态 | N | InMemoryHistory + JSONL | 是 |
| 附件状态 | N | dict[str, ContentPart] | 否 |
| 补全状态 | 3 | Completer 对象 | 否 |
| 键盘状态 | 7 | KeyBindings | 否 |

### 4.4.2 历史状态管理（事件溯源）

```python
# src/kimi_cli/ui/shell/prompt.py:320-359
class _HistoryEntry(BaseModel):
    content: str  # 不可变事件

def _load_history_entries(history_file: Path) -> list[_HistoryEntry]:
    """
    事件溯源模式：
    - 每个历史条目是不可变事件
    - 通过重放事件重建状态
    - 容错处理：跳过无效事件
    """
    entries: list[_HistoryEntry] = []
    if not history_file.exists():
        return entries  # 空状态
    
    try:
        with history_file.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue  # 跳过空事件
                
                try:
                    record = json.loads(line)  # 事件反序列化
                except json.JSONDecodeError:
                    logger.warning("Failed to parse; skipping: {line}")
                    continue  # 跳过无效事件
                
                try:
                    entry = _HistoryEntry.model_validate(record)
                    entries.append(entry)  # 应用有效事件
                except ValidationError:
                    logger.warning("Failed to validate; skipping: {line}")
                    continue  # 跳过无效事件
    except OSError as exc:
        logger.warning("Failed to load history file: {file} ({error})")
    
    return entries  # 重建的状态
```

**事件溯源模式**：
- **不可变事件**：`_HistoryEntry` 只包含 `content`，不可修改
- **重放重建**：通过重放所有事件重建当前状态
- **容错性**：跳过无效事件，不中断整个流程
- **可审计**：历史文件是完整的操作日志

**对比传统状态存储**：
```python
# 传统模式（存储最终状态）
{"history": ["cmd1", "cmd2", "cmd3"]}

# 事件溯源（存储事件序列）
{"content": "cmd1"}
{"content": "cmd2"}
{"content": "cmd3"}
```

**优势**：
- **时间旅行**：可以回溯到任意历史点
- **调试友好**：完整的历史记录
- **并发安全**：追加写入，无竞争条件
- **容错性**：单个事件损坏不影响其他事件

### 4.4.3 自动补全状态机

```python
# src/kimi_cli/ui/shell/prompt.py:91-196
class FileMentionCompleter(Completer):
    """
    复杂状态管理：
    - 缓存状态：_cached_paths, _cache_time
    - 分层缓存：顶层 vs 深层
    - 智能刷新：基于时间和输入
    - 忽略规则：多维度过滤
    """
    
    def __init__(self, root: Path, *, refresh_interval: float = 2.0, limit: int = 1000):
        self._root = root  # 不可变配置
        self._refresh_interval = refresh_interval  # 不可变配置
        self._limit = limit  # 不可变配置
        
        # 可变状态：缓存
        self._cache_time: float = 0.0
        self._cached_paths: list[str] = []
        self._top_cache_time: float = 0.0
        self._top_cached_paths: list[str] = []
        self._fragment_hint: str | None = None  # 临时状态
    
    def _get_paths(self) -> list[str]:
        """
        智能路由：根据输入片段选择缓存策略
        """
        fragment = self._fragment_hint or ""
        if "/" not in fragment and len(fragment) < 3:
            return self._get_top_level_paths()  # 浅层缓存
        return self._get_deep_paths()  # 深层缓存
    
    def _get_top_level_paths(self) -> list[str]:
        """
        缓存策略：时间窗口 + 容量限制
        """
        now = time.monotonic()
        if now - self._top_cache_time <= self._refresh_interval:
            return self._top_cached_paths  # 返回缓存状态
        
        # 缓存失效：重建状态
        entries: list[str] = []
        for entry in sorted(self._root.iterdir(), key=lambda p: p.name):
            if self._is_ignored(entry.name):
                continue  # 过滤状态
            entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)
            if len(entries) >= self._limit:
                break  # 容量限制
        
        # 更新缓存状态
        self._top_cached_paths = entries
        self._top_cache_time = now
        return entries
```

**状态机图**：
```
          ┌─────────────┐
          │   No Cache  │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │Check Time   │
          └──────┬──────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │Fresh ││Stale ││Stale │
    │      ││Top   ││Deep  │
    └──┬───┘ └──┬───┘ └──┬───┘
       │        │        │
       │        ▼        │
       │   ┌──────────┐  │
       │   │Rebuild   │  │
       │   └──────────┘  │
       └──────┬───────────┘
              │
              ▼
          ┌─────────┐
          │Cached   │
          └─────────┘
```

**缓存失效策略**：
- **时间窗口**：2 秒刷新间隔（平衡实时性与性能）
- **容量限制**：1000 条路径（防止内存爆炸）
- **智能路由**：根据输入选择缓存层级
- **惰性加载**：只在需要时重建缓存

### 4.4.4 键盘状态管理

```python
# src/kimi_cli/ui/shell/prompt.py:440-500
_kb = KeyBindings()

@_kb.add("enter", filter=has_completions)
def _accept_completion(event: KeyPressEvent) -> None:
    """
    条件状态：仅在 has_completions 时激活
    """
    buff = event.current_buffer
    if buff.complete_state and buff.complete_state.completions:
        completion = buff.complete_state.current_completion or buff.complete_state.completions[0]
        buff.apply_completion(completion)  # 状态变更：应用补全

@_kb.add("c-x", eager=True)
def _switch_mode(event: KeyPressEvent) -> None:
    """
    全局状态：Ctrl-X 总是有效（eager=True）
    """
    self._mode = self._mode.toggle()  # 状态转移：AGENT ↔ SHELL
    self._apply_mode(event)
    event.app.invalidate()  # 状态通知：重绘 UI

@Condition
def is_agent_mode() -> bool:
    """
    动态状态：基于 self._mode 实时计算
    """
    return self._mode == PromptMode.AGENT

@_kb.add("tab", filter=~has_completions & is_agent_mode, eager=True)
def _switch_thinking(event: KeyPressEvent) -> None:
    """
    复合条件状态：~has_completions & is_agent_mode
    """
    if "thinking" not in self._model_capabilities:
        console.print("[yellow]Thinking mode not supported[/yellow]")
        return
    self._thinking = not self._thinking  # 状态翻转
    event.app.invalidate()
```

**状态组合模式**：
- **简单条件**：`has_completions`（内置过滤器）
- **否定条件**：`~has_completions`（逻辑非）
- **自定义条件**：`is_agent_mode`（动态计算）
- **复合条件**：`~has_completions & is_agent_mode`（逻辑与）

**状态绑定机制**：
```python
# filter 参数控制状态激活条件
_kb.add("tab", filter=~has_completions & is_agent_mode, eager=True)

# 等价于
if (~has_completions & is_agent_mode)(event):
    if event.key == "tab":
        _switch_thinking(event)
```

## 4.5 异步任务协调模式

### 4.5.1 并行任务收集

```python
# src/kimi_cli/soul/runtime.py:79-82
ls_output, agents_md = await asyncio.gather(
    asyncio.to_thread(_list_work_dir, session.work_dir),
    asyncio.to_thread(load_agents_md, session.work_dir),
)
```

**模式**：`asyncio.gather()` 并行执行独立任务
- **状态一致性**：两个任务并行执行，结果同时返回
- **错误处理**：任一任务失败，整体失败（可通过 `return_exceptions=True` 修改）
- **类型安全**：返回结果顺序与输入顺序一致

### 4.5.2 异步上下文管理

```python
# src/kimi_cli/tools/mcp.py:22-25
async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnType:
    async with self._client as client:  # 异步上下文管理器
        result = await client.call_tool(self._mcp_tool.name, kwargs, timeout=20)
        return convert_tool_result(result)
```

**模式**：`async with` 确保资源正确清理
- **进入**：`__aenter__()` 建立连接
- **退出**：`__aexit__()` 关闭连接（即使异常）
- **状态保护**：确保资源状态不泄漏

### 4.5.3 异步生成器

```python
# src/kimi_cli/soul/kimisoul.py:212-219
result = await _kosong_step_with_retry()
if result.usage is not None:
    await self._context.update_token_count(result.usage.input)
    wire_send(StatusUpdate(status=self.status))

# 等待所有工具结果（异步生成器）
results = await result.tool_results()  # 可能产生多个结果
```

**模式**：异步生成器用于流式结果
- **惰性求值**：结果按需生成，不一次性加载
- **内存效率**：适合大量结果的场景
- **取消支持**：可在任意点取消生成

## 4.6 总结：异步状态管理最佳实践

### 4.6.1 状态保护模式

```python
# 模式 1：asyncio.shield() 保护关键状态更新
await asyncio.shield(self._grow_context(result, results))

# 模式 2：异常作为控制流
try:
    finished = await self._step()
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)
    continue

# 模式 3：任务清理
try:
    approval_task = asyncio.create_task(_pipe_approval_to_wire())
    ...
finally:
    approval_task.cancel()
```

### 4.6.2 状态分层原则

```
全局状态（yolo）
    ↓
会话状态（auto_approve_actions）
    ↓
请求状态（ApprovalRequest）
    ↓
临时状态（_fragment_hint）
```

### 4.6.3 异步通信模式

```python
# 模式 1：Queue（生产者-消费者）
request_queue = asyncio.Queue[ApprovalRequest]()
await queue.put(request)
request = await queue.get()

# 模式 2：Event（信号）
event = asyncio.Event()
event.set()
await event.wait()

# 模式 3：Condition（条件变量）
cond = asyncio.Condition()
async with cond:
    await cond.wait()
    cond.notify_all()
```

### 4.6.4 性能优化模式

```python
# 模式 1：缓存 + 失效策略
if now - cache_time <= refresh_interval:
    return cached_paths  # 快速路径
# 重建缓存（慢速路径）

# 模式 2：惰性加载
if "/" not in fragment and len(fragment) < 3:
    return self._get_top_level_paths()  # 浅层
return self._get_deep_paths()  # 深层

# 模式 3：批量操作
results = await asyncio.gather(*tasks)  # 并行执行
```

---

# 专题 5: UI 多模式架构与交互设计

## 5.1 架构概览

Kimi CLI 支持三种 UI 模式：
- **Shell**：交互式终端（默认，724 行代码）
- **Print**：非交互式脚本模式
- **ACP**：Agent Client Protocol 服务器模式

**状态管理挑战**：
- 模式切换时的状态一致性
- 不同模式的状态隔离
- 共享状态的管理

## 5.2 Shell 模式 - 复杂交互状态机

### 5.2.1 双模式设计（Agent vs Shell）

```python
# src/kimi_cli/ui/shell/prompt.py:362-370
class PromptMode(Enum):
    AGENT = "agent"  # AI 助手模式
    SHELL = "shell"  # 命令行模式
    
    def toggle(self) -> "PromptMode":
        return PromptMode.SHELL if self == PromptMode.AGENT else PromptMode.AGENT
```

**状态维度分析**：

| 维度 | Agent 模式 | Shell 模式 |
|------|-----------|-----------|
| **Prompt 符号** | ✨ | $ |
| **Completer** | MetaCommand + FileMention | 系统默认 |
| **Thinking 支持** | 支持（Tab 切换） | 不支持 |
| **命令解释** | AI 解析 | Shell 直接执行 |

**切换状态流转**：
```python
# src/kimi_cli/ui/shell/prompt.py:455-460
@_kb.add("c-x", eager=True)
def _switch_mode(event: KeyPressEvent) -> None:
    self._mode = self._mode.toggle()  # 状态翻转
    self._apply_mode(event)  # 应用模式特定设置
    event.app.invalidate()  # 状态通知：重绘 UI
```

**状态一致性保证**：
```python
def _apply_mode(self, event: KeyPressEvent) -> None:
    """
    模式切换时的状态同步
    """
    if self._mode == PromptMode.AGENT:
        # Agent 模式：启用自定义 completer
        event.app.current_buffer.completer = self._agent_mode_completer
    else:
        # Shell 模式：禁用 completer（使用系统默认）
        event.app.current_buffer.completer = None
```

### 5.2.2 思考模式状态

```python
# src/kimi_cli/ui/shell/prompt.py:491-499
@_kb.add("tab", filter=~has_completions & is_agent_mode, eager=True)
def _switch_thinking(event: KeyPressEvent) -> None:
    """
    条件状态切换：
    - 仅在 Agent 模式（is_agent_mode）
    - 且无补全菜单（~has_completions）
    - 且模型支持 thinking 能力
    """
    if "thinking" not in self._model_capabilities:
        console.print("[yellow]Thinking mode not supported[/yellow]")
        return  # 状态不变
    
    self._thinking = not self._thinking  # 状态翻转
    event.app.invalidate()  # 状态通知
```

**状态联动**：
```python
@property
def prompt_symbol(self) -> str:
    """
    派生状态：基于多个状态计算 Prompt 符号
    """
    if self._mode == PromptMode.SHELL:
        return PROMPT_SYMBOL_SHELL
    if self._thinking:
        return PROMPT_SYMBOL_THINKING
    return PROMPT_SYMBOL
```

**状态转换图**：
```
          ┌─────────────┐
          │Agent Mode   │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │Check Caps   │
          └──────┬──────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │No    ││Yes   ││No    │
    │Support││Support││Thinking│
    └──┬───┘ └──┬───┘ └──┬───┘
       │        │        │
       │        ▼        │
       │   ┌──────────┐  │
       │   │Toggle    │  │
       │   └──────────┘  │
       └──────┬───────────┘
              │
              ▼
          ┌─────────┐
          │Thinking │
          │Toggled  │
          └─────────┘
```

### 5.2.3 附件状态管理

```python
# src/kimi_cli/ui/shell/prompt.py:418
self._attachment_parts: dict[str, ContentPart] = {}
"""Mapping from attachment id to ContentPart."""

# src/kimi_cli/ui/shell/prompt.py:472-476
@_kb.add("c-v", eager=True)
def _paste(event: KeyPressEvent) -> None:
    if self._try_paste_image(event):
        return  # 图片已处理
    clipboard_data = event.app.clipboard.get_data()
    event.current_buffer.paste_clipboard_data(clipboard_data)
```

**状态生命周期**：
```
粘贴图片
    ↓
生成 attachment_id（随机字符串）
    ↓
存储到 _attachment_parts[attachment_id]
    ↓
在输入中插入占位符 [image:attachment_id]
    ↓
提交时解析占位符 → 替换为实际 ContentPart
    ↓
清理 _attachment_parts（可选）
```

**占位符解析**：
```python
# src/kimi_cli/ui/shell/prompt.py:397-399
_ATTACHMENT_PLACEHOLDER_RE = re.compile(
    r"\[(?P<type>image):(?P<id>[a-zA-Z0-9_\-\.]+)(?:,(?P<width>\d+)x(?P<height>\d+))?\]"
)
```

**状态转换**：
```python
def _parse_attachments(self, text: str) -> list[ContentPart]:
    """
    文本状态 → 富文本状态
    """
    parts: list[ContentPart] = []
    last_end = 0
    
    for match in _ATTACHMENT_PLACEHOLDER_RE.finditer(text):
        # 添加文本部分
        if match.start() > last_end:
            parts.append(TextPart(text[last_end:match.start()]))
        
        # 添加附件部分
        attachment_id = match.group("id")
        if attachment_id in self._attachment_parts:
            parts.append(self._attachment_parts[attachment_id])
        
        last_end = match.end()
    
    # 剩余文本
    if last_end < len(text):
        parts.append(TextPart(text[last_end:]))
    
    return parts
```

## 5.3 Print 模式 - 无状态设计

```python
# src/kimi_cli/ui/print/__init__.py
class PrintApp:
    """
    无状态设计哲学：
    - 不存储任何会话状态
    - 输入 → 处理 → 输出 → 退出
    - 适合脚本和 CI/CD
    """
    
    def __init__(self, soul: KimiSoul):
        self._soul = soul  # 只读依赖
    
    async def run(self, user_input: str) -> int:
        """
        纯函数式：无副作用（除了日志和输出）
        """
        try:
            await self._soul.run(user_input)
            return 0  # 成功状态码
        except Exception as e:
            logger.error("Error: {error}", error=e)
            return 1  # 错误状态码
```

**设计权衡**：

| 维度 | Shell 模式 | Print 模式 |
|------|-----------|-----------|
| **状态管理** | 复杂（724 行） | 简单（~50 行） |
| **用户体验** | 交互式 | 批处理 |
| **适用场景** | 终端用户 | 脚本/CI |
| **状态持久化** | 历史、checkpoint | 无 |
| **资源占用** | 高（长期运行） | 低（一次性） |

## 5.4 ACP 模式 - 协议状态管理

```python
# src/kimi_cli/ui/acp/__init__.py
class ACPApp:
    """
    Agent Client Protocol 服务器模式
    
    状态管理挑战：
    - 多客户端并发
    - 长连接状态保持
    - 协议一致性
    """
    
    def __init__(self, soul_factory: Callable[[], KimiSoul]):
        self._soul_factory = soul_factory  # 状态工厂（每个请求独立）
        self._sessions: dict[str, KimiSoul] = {}  # 会话状态（可选）
    
    async def handle_request(self, request: dict) -> dict:
        """
        无状态请求处理：
        - 每个请求创建新的 Soul 实例
        - 请求处理完销毁状态
        - 或可选的会话保持
        """
        session_id = request.get("session_id")
        
        if session_id and session_id in self._sessions:
            soul = self._sessions[session_id]  # 复用会话状态
        else:
            soul = self._soul_factory()  # 创建新状态
            if session_id:
                self._sessions[session_id] = soul  # 存储会话状态
        
        try:
            result = await soul.run(request["input"])
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

**状态策略选择**：
```python
# 策略 1：无状态（简单，可扩展）
soul = self._soul_factory()
result = await soul.run(input)

# 策略 2：会话状态（复杂，有状态）
if session_id not in self._sessions:
    self._sessions[session_id] = self._soul_factory()
result = await self._sessions[session_id].run(input)
```

**权衡分析**：

| 策略 | 优势 | 代价 | 适用场景 |
|------|------|------|---------|
| **无状态** | 简单、可扩展、容错 | 无历史、无上下文 | API 服务 |
| **会话状态** | 有历史、有上下文 | 复杂、资源占用、会话管理 | 聊天应用 |

## 5.5 跨模式状态共享

### 5.5.1 共享状态设计

```python
# src/kimi_cli/soul/runtime.py:62-96
class Runtime(NamedTuple):
    """
    跨模式共享状态：
    - 配置状态（config）
    - LLM 状态（llm）
    - 会话状态（session）
    - 内置参数（builtin_args）
    - 通信中心（denwa_renji）
    - 审批系统（approval）
    """
    config: Config
    llm: LLM | None
    session: Session
    builtin_args: BuiltinSystemPromptArgs
    denwa_renji: DenwaRenji
    approval: Approval
```

**状态共享策略**：
- **不可变状态**：`NamedTuple` 确保配置状态不被修改
- **依赖注入**：通过构造函数传递共享状态
- **接口隔离**：每个模式只依赖需要的部分

### 5.5.2 模式特定状态

```python
# Shell 模式特定状态
class ShellApp:
    def __init__(self, runtime: Runtime):
        self._runtime = runtime  # 共享状态
        self._prompt_session = CustomPromptSession(...)  # 私有状态
        self._soul = KimiSoul(...)  # 私有状态

# Print 模式特定状态
class PrintApp:
    def __init__(self, soul: KimiSoul):
        self._soul = soul  # 只读共享状态
        # 无私有状态（无状态设计）

# ACP 模式特定状态
class ACPApp:
    def __init__(self, soul_factory: Callable[[], KimiSoul]):
        self._soul_factory = soul_factory  # 状态工厂
        self._sessions = {}  # 私有状态：会话管理
```

**状态隔离原则**：
```
共享状态（Runtime）
    ├── Shell 私有状态（PromptSession, Soul）
    ├── Print 私有状态（无）
    └── ACP 私有状态（Sessions）
```

## 5.6 总结：UI 状态管理最佳实践

### 5.6.1 模式切换原则

```python
# 原则 1：原子切换
self._mode = self._mode.toggle()  # 状态变更
self._apply_mode(event)  # 应用变更
event.app.invalidate()  # 通知变更

# 原则 2：条件激活
@_kb.add("tab", filter=~has_completions & is_agent_mode)

# 原则 3：派生状态
@property
def prompt_symbol(self) -> str:
    return PROMPT_SYMBOL_SHELL if self._mode == PromptMode.SHELL else ...
```

### 5.6.2 状态分层

```
全局状态（Runtime）
    ↓
模式状态（PromptMode）
    ↓
功能状态（Thinking, Attachments）
    ↓
临时状态（Completions, Keyboard）
```

### 5.6.3 性能优化

```python
# 缓存策略
if now - cache_time <= refresh_interval:
    return cached_paths

# 惰性加载
if "/" not in fragment and len(fragment) < 3:
    return self._get_top_level_paths()

# 容量限制
if len(entries) >= self._limit:
    break
```

---

# 专题 6: 测试策略与代码质量保障

## 6.1 测试架构概览

```
tests/
├── conftest.py              # 共享 Fixture（80 行）
├── test_*.py               # 26 个测试文件
└── 测试策略：
    ├── 单元测试（工具函数）
    ├── 集成测试（Agent 工作流）
    ├── 异步测试（asyncio 支持）
    └── Mock 策略（LLM 调用）
```

**测试金字塔**：

```
      ┌─────────────┐
      │  E2E 测试   │  (tests_ai/)
      └──────┬──────┘
             │
      ┌─────────────┐
      │ 集成测试    │  (test_task_subagents.py)
      └──────┬──────┘
             │
      ┌─────────────┐
      │  单元测试   │  (test_*.py)
      └─────────────┘
```

## 6.2 Fixture 设计模式

### 6.2.1 共享 Fixture 架构

```python
# tests/conftest.py:34-77
@pytest.fixture
def config() -> Config:
    """
    Fixture 模式：工厂函数
    - 创建并配置测试对象
    - 测试用例通过参数注入
    - 每个测试独立实例（函数作用域）
    """
    conf = get_default_config()
    conf.services.moonshot_search = MoonshotSearchConfig(
        base_url="https://api.kimi.com/coding/v1/search",
        api_key=SecretStr("test-api-key"),  # 测试专用密钥
    )
    return conf

@pytest.fixture
def llm() -> LLM:
    """
    Fixture 模式：Mock 对象
    - 使用 MockChatProvider 避免真实 LLM 调用
    - 确定性行为（测试可重复）
    - 快速执行（无网络延迟）
    """
    return LLM(
        chat_provider=MockChatProvider([]),  # 空响应
        max_context_size=100_000,
        capabilities=set(),
    )

@pytest.fixture
def temp_work_dir() -> Generator[Path]:
    """
    Fixture 模式：临时资源管理
    - 创建临时目录（setUp）
    - 自动清理（tearDown）
    - 异常安全（即使测试失败也清理）
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)  # 提供给测试用例
    # 退出时自动删除

@pytest.fixture
def builtin_args(temp_work_dir: Path) -> BuiltinSystemPromptArgs:
    """
    Fixture 模式：依赖其他 Fixture
    - 自动注入 temp_work_dir
    - 构建派生对象
    - 类型安全
    """
    return BuiltinSystemPromptArgs(
        KIMI_NOW="1970-01-01T00:00:00+00:00",  # 固定时间（确定性）
        KIMI_WORK_DIR=temp_work_dir,
        KIMI_WORK_DIR_LS="Test ls content",
        KIMI_AGENTS_MD="Test agents content",
    )
```

**Fixture 依赖图**：
```
builtin_args
    └── temp_work_dir

runtime
    ├── config
    ├── llm
    ├── session
    ├── builtin_args
    ├── denwa_renji
    └── approval

agent_spec
    └── (文件系统)
```

### 6.2.2 Fixture 作用域策略

```python
# tests/conftest.py:80-107
@pytest.fixture
def runtime(config, llm, session, builtin_args) -> Runtime:
    """
    作用域：函数级（默认）
    - 每个测试函数独立实例
    - 测试间状态隔离
    - 避免测试相互影响
    """
    return Runtime(
        config=config,
        llm=llm,
        session=session,
        builtin_args=builtin_args,
        denwa_renji=DenwaRenji(),  # 全新实例
        approval=Approval(yolo=False),
    )

# 模块级 Fixture（共享状态）
@pytest.fixture(scope="module")
def shared_resource():
    """
    作用域：模块级
    - 同一模块内测试共享
    - 减少重复初始化
    - 需要谨慎处理状态清理
    """
    resource = ExpensiveResource()
    yield resource
    resource.cleanup()

# 会话级 Fixture（全局共享）
@pytest.fixture(scope="session")
def global_config():
    """
    作用域：会话级
    - 所有测试共享
    - 只初始化一次
    - 适合不可变配置
    """
    return ImmutableConfig()
```

**作用域选择指南**：

| 作用域 | 创建次数 | 适用场景 | 风险 |
|--------|---------|---------|------|
| **function** | 每个测试 1 次 | 大多数情况 | 无 |
| **module** | 每个模块 1 次 | 昂贵资源 | 状态泄漏 |
| **session** | 整个会话 1 次 | 全局配置 | 不可变数据 |

## 6.3 测试模式

### 6.3.1 单元测试模式

```python
# tests/test_agent_spec.py:1-50
def test_load_agent_spec():
    """
    单元测试模式：孤立测试
    - 测试单一函数
    - Mock 外部依赖
    - 快速执行
    """
    agent_file = Path("tests/fixtures/test_agent.yaml")
    spec = load_agent_spec(agent_file)
    
    assert spec.name == "test-agent"
    assert spec.tools == ["bash", "file"]
    assert spec.system_prompt_path.exists()

def test_load_agent_spec_not_found():
    """
    单元测试模式：异常路径
    - 测试错误处理
    - 验证异常类型和消息
    """
    agent_file = Path("nonexistent.yaml")
    with pytest.raises(FileNotFoundError):
        load_agent_spec(agent_file)

def test_load_agent_spec_invalid():
    """
    单元测试模式：无效输入
    - 测试验证逻辑
    - 确保健壮性
    """
    agent_file = Path("tests/fixtures/invalid_agent.yaml")
    with pytest.raises(AgentSpecError):
        load_agent_spec(agent_file)
```

**测试金字塔底层**：
- **数量**：最多（70% 测试）
- **速度**：最快（毫秒级）
- **维护成本**：低
- **ROI**：最高

### 6.3.2 集成测试模式

```python
# tests/test_task_subagents.py:1-100
@pytest.mark.asyncio
async def test_task_subagent_delegation(runtime: Runtime):
    """
    集成测试模式：组件协作
    - 测试多个组件交互
    - 使用真实依赖（Fixture）
    - 验证端到端流程
    """
    # 创建主 Agent
    main_agent = Agent(...)
    soul = KimiSoul(main_agent, runtime, context=Context(...))
    
    # 执行包含子任务的用户输入
    await soul.run("Create a new Python file with a function")
    
    # 验证子 Agent 被调用
    assert "Task" in [tool.name for tool in main_agent.toolset.tools]
    # 验证文件被创建
    assert (runtime.session.work_dir / "new_file.py").exists()

@pytest.mark.asyncio
async def test_task_subagent_error_handling(runtime: Runtime):
    """
    集成测试模式：错误处理
    - 测试异常传播
    - 验证恢复机制
    """
    soul = KimiSoul(...)
    
    # 子任务失败
    with pytest.raises(ToolRejectedError):
        await soul.run("Execute dangerous command")
```

**测试金字塔中层**：
- **数量**：中等（20% 测试）
- **速度**：中等（秒级）
- **维护成本**：中等
- **ROI**：高（发现组件集成问题）

### 6.3.3 异步测试模式

```python
# tests/test_bash.py:1-80
@pytest.mark.asyncio
async def test_bash_success():
    """
    异步测试模式：async/await
    - 使用 pytest-asyncio
    - 支持异步 Fixture
    - 异步断言
    """
    bash = Bash()
    result = await bash("echo 'hello world'")
    
    assert result.exit_code == 0
    assert "hello world" in result.stdout

@pytest.mark.asyncio
async def test_bash_timeout():
    """
    异步测试模式：超时测试
    - 测试异步超时机制
    - 验证异常类型
    """
    bash = Bash(timeout=0.1)
    
    with pytest.raises(asyncio.TimeoutError):
        await bash("sleep 10")

@pytest.mark.asyncio
async def test_bash_concurrent():
    """
    异步测试模式：并发测试
    - 测试并发执行
    - 验证隔离性
    """
    bash = Bash()
    
    # 并发执行多个命令
    results = await asyncio.gather(
        bash("echo 'cmd1'"),
        bash("echo 'cmd2'"),
        bash("echo 'cmd3'"),
    )
    
    assert len(results) == 3
    assert all(r.exit_code == 0 for r in results)
```

**异步测试最佳实践**：

1. **标记测试**：`@pytest.mark.asyncio`
2. **异步 Fixture**：Fixture 可以是 async 函数
3. **超时控制**：`pytest-timeout` 插件
4. **隔离性**：每个测试独立事件循环

### 6.3.4 Mock 策略

```python
# tests/conftest.py:45-52
@pytest.fixture
def llm() -> LLM:
    """
    Mock 策略：替换外部依赖
    - 避免真实 LLM 调用（成本高、慢、不稳定）
    - 提供确定性响应
    - 控制测试场景
    """
    return LLM(
        chat_provider=MockChatProvider([
            # 预定义响应序列
            "I'll help you create a file",
            "The file has been created successfully",
        ]),
        max_context_size=100_000,
        capabilities=set(),
    )

# tests/test_soul_message.py:1-50
def test_tool_result_to_messages():
    """
    Mock 策略：函数级别 Mock
    - 使用 unittest.mock
    - 验证调用次数和参数
    """
    from unittest.mock import Mock, patch
    
    mock_tool = Mock()
    mock_tool.name = "TestTool"
    mock_tool.result = "Test result"
    
    messages = tool_result_to_messages(mock_tool)
    
    assert len(messages) == 1
    assert messages[0].role == "tool"
    assert "Test result" in messages[0].content
```

**Mock 层次**：

```
LLM API 调用（最高层）
    ↓
ChatProvider（高层）
    ↓
HTTP Client（低层）
    ↓
Socket（最底层）
```

**Mock 策略选择**：

| 层次 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **LLM API** | 简单、控制力强 | 离实现远 | 集成测试 |
| **ChatProvider** | 平衡、真实 | 需要理解接口 | 大多数测试 |
| **HTTP Client** | 真实、全面 | 复杂、脆弱 | 网络层测试 |

## 6.4 测试质量保障

### 6.4.1 覆盖率策略

```bash
# 运行测试并生成覆盖率报告
uv run pytest tests --cov=kimi_cli --cov-report=html

# 关键指标：
# - 总体覆盖率 > 80%
# - 核心模块 > 90%（soul/, tools/）
# - 异常路径必须覆盖
```

**覆盖率目标**：

| 模块 | 目标 | 重要性 | 策略 |
|------|------|--------|------|
| soul/ | 90% | 核心 | 必须覆盖 |
| tools/ | 85% | 重要 | 必须覆盖 |
| agentspec.py | 90% | 重要 | 必须覆盖 |
| ui/ | 70% | 中等 | 优先覆盖关键路径 |
| cli.py | 60% | 低 | 集成测试覆盖 |

### 6.4.2 测试数据管理

```python
# tests/fixtures/
fixtures/
├── agents/
│   ├── test_agent.yaml      # 有效配置
│   ├── invalid_agent.yaml   # 无效配置
│   └── extended_agent.yaml  # 继承配置
├── files/
│   ├── test_file.txt        # 测试文件
│   └── binary_file.bin      # 二进制文件
└── history/
    └── sample_history.jsonl # 历史数据
```

**测试数据原则**：
- **确定性**：测试数据不变，结果可预测
- **隔离性**：每个测试独立数据
- **代表性**：覆盖正常、边界、异常场景
- **版本化**：测试数据纳入版本控制

### 6.4.3 测试执行策略

```bash
# 快速测试（开发时）
uv run pytest tests -x  # 首次失败停止

# 完整测试（提交前）
uv run pytest tests -v --cov=kimi_cli

# 并行测试（加速）
uv run pytest tests -n auto  # pytest-xdist

# 随机测试（发现依赖）
uv run pytest tests --random-order  # pytest-random-order
```

**CI/CD 集成**：
```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: uv run pytest tests --cov=kimi_cli --cov-fail-under=80
  
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## 6.5 代码质量工具链

### 6.5.1 静态分析

```bash
# 类型检查
uv run pyright src/kimi_cli

# Linting
uv run ruff check src/kimi_cli

# 格式化
uv run ruff format src/kimi_cli

# 拼写检查
uv run typos
```

**工具配置**（pyproject.toml）：
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "UP", "B", "SIM", "I"]

[tool.pyright]
strict = ["src/kimi_cli/soul"]
reportUnknownMemberType = "error"
```

### 6.5.2 预提交钩子

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: uv run pyright
        language: system
        types: [python]
```

**预提交工作流**：
```bash
# 安装钩子
git config core.hooksPath .githooks

# 提交时自动执行
1. ruff check --fix
2. ruff format
3. pyright
4. pytest (快速测试)
```

## 6.6 总结：测试与质量保障最佳实践

### 6.6.1 测试金字塔

```
      ┌─────────────────┐
      │   E2E (1%)      │  Kimi CLI 自动测试
      └────────┬────────┘
               │
      ┌────────┴────────┐
      │ 集成 (20%)      │  测试工作流
      └────────┬────────┘
               │
      ┌────────┴────────┐
      │ 单元 (70%)      │  测试函数
      └────────┬────────┘
               │
      ┌────────┴────────┐
      │ 静态分析 (9%)   │  类型、Lint
      └─────────────────┘
```

### 6.6.2 测试原则

1. **FIRST 原则**：
   - **F**ast（快速）
   - **I**solated（隔离）
   - **R**epeatable（可重复）
   - **S**elf-verifying（自验证）
   - **T**imely（及时）

2. **Arrange-Act-Assert**：
   ```python
   def test_example():
       # Arrange: 准备数据
       data = setup()
       
       # Act: 执行操作
       result = function_under_test(data)
       
       # Assert: 验证结果
       assert result == expected
   ```

3. **一个测试一个断言**（理想状态）

### 6.6.3 质量门禁

```bash
# 提交前检查清单
□ uv run ruff check --fix
□ uv run ruff format --check
□ uv run pyright
□ uv run pytest tests -x
□ uv run pytest tests --cov=kimi_cli --cov-fail-under=80
```

**自动化**：
- **本地**：预提交钩子
- **CI**：GitHub Actions
- **CD**：覆盖率门禁

---

# 总结：Kimi CLI 架构精髓

## 7.1 状态管理核心原则

### 7.1.1 分层状态

```
配置状态（不可变）
    ↓
会话状态（可回溯）
    ↓
请求状态（临时）
    ↓
临时状态（用完即弃）
```

### 7.1.2 状态持久化模式

```python
# 模式 1：事件溯源（Context）
{"role": "user", "content": "..."}
{"role": "_checkpoint", "id": 1}

# 模式 2：双写（内存 + 磁盘）
self._history.append(message)
await f.write(json.dumps(...))

# 模式 3：文件轮转（历史保留）
mv context.json context.json.1
```

### 7.1.3 状态回溯机制

```python
# 异常作为控制流
try:
    finished = await self._step()
except BackToTheFuture as e:
    await self._context.revert_to(e.checkpoint_id)
    continue
```

## 7.2 异步编程精髓

### 7.2.1 并发模式

```python
# 并行收集
data1, data2 = await asyncio.gather(task1, task2)

# 生产者-消费者
queue = asyncio.Queue()
await queue.put(item)
item = await queue.get()

# 保护关键操作
await asyncio.shield(critical_operation())
```

### 7.2.2 任务生命周期

```python
# 创建
task = asyncio.create_task(coro())

# 取消
task.cancel()

# 等待
try:
    await task
except asyncio.CancelledError:
    ...
```

## 7.3 设计模式应用

### 7.3.1 创建型

```python
# 工厂模式
soul_factory = lambda: KimiSoul(...)

# 构建器模式
Runtime.create(config, llm, session, yolo)
```

### 7.3.2 结构型

```python
# 适配器模式
class MCPTool(CallableTool):
    def __init__(self, mcp_tool: mcp.Tool, ...):
        self._mcp_tool = mcp_tool  # 被适配者
    
    async def __call__(self, **kwargs):
        result = await self._client.call_tool(...)  # 委托
        return convert_tool_result(result)  # 转换

# 装饰器模式
@tenacity.retry(...)
async def _kosong_step_with_retry():
    ...
```

### 7.3.3 行为型

```python
# 策略模式
class SimpleCompaction(Compaction):
    async def compact(self, messages, llm):
        ...  # 具体策略

# 状态模式
class Context:
    def checkpoint(self):
        ...  # 状态转移
    
    def revert_to(self, checkpoint_id):
        ...  # 状态回溯

# 观察者模式
wire_send(StepBegin(step_no))
wire_send(StatusUpdate(status))
```

## 7.4 测试哲学

### 7.4.1 测试金字塔

```
      ┌─────────────────┐
      │   E2E (1%)      │
      └────────┬────────┘
             │
      ┌────────┴────────┐
      │ 集成 (20%)      │
      └────────┬────────┘
             │
      ┌────────┴────────┐
      │  单元 (70%)     │
      └────────┬────────┘
             │
      ┌────────┴────────┐
      │ 静态分析 (9%)   │
      └─────────────────┘
```

### 7.4.2 FIRST 原则

- **F**ast（快速）
- **I**solated（隔离）
- **R**epeatable（可重复）
- **S**elf-verifying（自验证）
- **T**imely（及时）

## 7.5 可复用的架构模式

### 7.5.1 状态管理模板

```python
class StateManager:
    def __init__(self, file_backend: Path):
        self._file_backend = file_backend
        self._cache: list[State] = []
        self._token_count = 0
    
    async def restore(self) -> bool:
        if not self._file_backend.exists():
            return False
        async with aiofiles.open(...) as f:
            async for line in f:
                event = json.loads(line)
                self._apply_event(event)
        return True
    
    async def checkpoint(self):
        await self._append_event({"type": "checkpoint", "id": ...})
    
    async def revert_to(self, checkpoint_id: int):
        # 文件轮转 + 重放
        ...
```

### 7.5.2 异步任务模板

```python
async def _agent_loop(self):
    step_no = 1
    while True:
        try:
            await self._checkpoint()
            finished = await self._step()
        except BackToTheFuture as e:
            await self._context.revert_to(e.checkpoint_id)
            continue
        except (ChatProviderError, asyncio.CancelledError):
            raise
        
        if finished:
            return
        
        step_no += 1
        if step_no > self._loop_control.max_steps_per_run:
            raise MaxStepsReached(...)
```

### 7.5.3 审批系统模板

```python
class Approval:
    def __init__(self, auto_approve: bool = False):
        self._auto_approve = auto_approve
        self._approved_actions: set[str] = set()
        self._queue = asyncio.Queue[Request]()
    
    async def request(self, action: str) -> bool:
        if self._auto_approve or action in self._approved_actions:
            return True
        
        request = Request(action)
        await self._queue.put(request)
        response = await request.wait()
        
        if response.approve_for_session:
            self._approved_actions.add(action)
        
        return response.approved
    
    async def fetch_request(self) -> Request:
        return await self._queue.get()
```

## 7.6 下一步学习建议

### 7.6.1 实践项目

1. **添加新工具**：
   ```python
   # 1. 创建工具模块
   # src/kimi_cli/tools/new_tool/__init__.py
   
   # 2. 注册工具
   # src/kimi_cli/tools/__init__.py
   TOOL_MODULES["new_tool"] = "kimi_cli.tools.new_tool"
   
   # 3. 编写测试
   # tests/test_new_tool.py
   
   # 4. 更新文档
   # AGENTS.md
   ```

2. **实现新 UI 模式**：
   ```python
   # src/kimi_cli/ui/new_mode/__init__.py
   class NewModeApp:
       def __init__(self, soul: KimiSoul):
           self._soul = soul
       
       async def run(self) -> int:
           ...
   ```

3. **优化状态压缩**：
   ```python
   # src/kimi_cli/soul/compaction.py
   class SmartCompaction(Compaction):
       async def compact(self, messages, llm):
           # 基于重要性评分
           # 选择性保留关键消息
           ...
   ```

### 7.6.2 深入研究

1. **阅读源码**：
   - `src/kimi_cli/soul/kimisoul.py`（核心循环）
   - `src/kimi_cli/ui/shell/prompt.py`（交互状态）
   - `src/kimi_cli/tools/mcp.py`（协议适配）

2. **学习依赖库**：
   - `kosong`（LLM 框架）
   - `prompt-toolkit`（终端 UI）
   - `fastmcp`（MCP 协议）

3. **架构模式**：
   - 事件溯源
   - CQRS（命令查询职责分离）
   - Actor 模型

---

**文档完成！** 共 6 个专题，约 2000 行，涵盖 Kimi CLI 的核心架构与设计哲学。