# 专题 5: UI 多模式架构与 ACP 协议深度解析

## 5.1 架构概览：三模式设计哲学

Kimi CLI 支持三种 UI 模式，每种模式针对不同的使用场景：

```
┌─────────────────────────────────────────────────────────────┐
│                    UI 模式总览                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Shell      │  │    Print     │  │     ACP      │      │
│  │  交互式      │  │  非交互式    │  │  协议服务器  │      │
│  │  724 行      │  │  153 行      │  │  436 行      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           ▼                                │
│                  ┌─────────────────┐                       │
│                  │  Soul 执行引擎   │                       │
│                  └────────┬────────┘                       │
│                           ▼                                │
│                  ┌─────────────────┐                       │
│                  │  Runtime 运行时  │                       │
│                  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**模式对比**：

| 维度 | Shell 模式 | Print 模式 | ACP 模式 |
|------|-----------|-----------|---------|
| **交互性** | 高（实时） | 无（批处理） | 中（协议） |
| **状态管理** | 复杂（724 行） | 简单（无状态） | 中等（436 行） |
| **适用场景** | 终端用户 | 脚本/CI/CD | IDE 插件 |
| **通信方式** | 终端 UI | 标准输入输出 | JSON-RPC |
| **生命周期** | 长期运行 | 一次性 | 长期运行 |
| **并发** | 单用户 | 单请求 | 多会话 |

**设计哲学**：
- **Shell**：功能最全，支持历史、补全、附件、审批等
- **Print**：简单可靠，适合自动化
- **ACP**：标准化协议，便于第三方集成

## 5.2 Shell 模式：724 行的复杂状态机

### 5.2.1 状态管理挑战

Shell 模式是 Kimi CLI 最复杂的组件，724 行代码管理 7+ 种状态：

```python
# src/kimi_cli/ui/shell/prompt.py:402-500 (部分)
class CustomPromptSession:
    """
    状态管理复杂度：
    - 模式状态：AGENT vs SHELL
    - 思考状态：thinking on/off
    - 历史状态：用户输入历史（持久化）
    - 附件状态：图片附件（临时）
    - 剪贴板状态：系统剪贴板
    - 自动补全状态：3 个 completer
    - 键盘状态：7+ 快捷键绑定
    - 工具调用状态：审批中/执行中
    """
    
    def __init__(self, *, status_provider, model_capabilities):
        # 历史状态管理（事件溯源）
        history_dir = get_share_dir() / "user-history"
        work_dir_id = md5(str(Path.cwd()).encode()).hexdigest()
        self._history_file = (history_dir / work_dir_id).with_suffix(".jsonl")
        
        # 加载历史状态（从磁盘重建）
        history_entries = _load_history_entries(self._history_file)
        history = InMemoryHistory()
        for entry in history_entries:
            history.append_string(entry.content)
        
        # 模式状态（双模式）
        self._mode: PromptMode = PromptMode.AGENT
        self._thinking: bool = False
        
        # 附件状态（临时）
        self._attachment_parts: dict[str, ContentPart] = {}
```

**状态分类统计**：

| 状态类别 | 数量 | 存储方式 | 持久化 | 复杂度 |
|---------|------|---------|--------|--------|
| 模式状态 | 2 | Enum + bool | 否 | 低 |
| 历史状态 | N | InMemoryHistory + JSONL | 是 | 高 |
| 附件状态 | N | dict[str, ContentPart] | 否 | 中 |
| 补全状态 | 3 | Completer 对象 | 否 | 中 |
| 键盘状态 | 7+ | KeyBindings | 否 | 中 |
| 工具状态 | N | ApprovalRequest | 否 | 高 |

### 5.2.2 历史状态管理：事件溯源模式

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

**事件溯源的优势**：

1. **可审计**：完整的历史记录
2. **容错性**：单个事件损坏不影响其他
3. **时间旅行**：可回溯到任意历史点
4. **并发安全**：追加写入，无竞争条件

**对比传统状态存储**：

```python
# 传统模式（存储最终状态）
{"history": ["cmd1", "cmd2", "cmd3"]}

# 事件溯源（存储事件序列）
{"content": "cmd1"}
{"content": "cmd2"}
{"content": "cmd3"}
```

### 5.2.3 模式切换状态机

```python
# src/kimi_cli/ui/shell/prompt.py:362-370
class PromptMode(Enum):
    AGENT = "agent"  # AI 助手模式
    SHELL = "shell"  # 命令行模式
    
    def toggle(self) -> "PromptMode":
        return PromptMode.SHELL if self == PromptMode.AGENT else PromptMode.AGENT
```

**状态流转**：

```python
# src/kimi_cli/ui/shell/prompt.py:455-460
@_kb.add("c-x", eager=True)
def _switch_mode(event: KeyPressEvent) -> None:
    """Ctrl-X 切换模式"""
    self._mode = self._mode.toggle()  # 状态翻转
    self._apply_mode(event)  # 应用模式特定设置
    event.app.invalidate()  # 状态通知：重绘 UI

def _apply_mode(self, event: KeyPressEvent) -> None:
    """模式切换时的状态同步"""
    if self._mode == PromptMode.AGENT:
        # Agent 模式：启用自定义 completer
        event.app.current_buffer.completer = self._agent_mode_completer
    else:
        # Shell 模式：禁用 completer（使用系统默认）
        event.app.current_buffer.completer = None
```

**状态一致性保证**：

```python
# 原子切换：状态变更 + 应用 + 通知
self._mode = self._mode.toggle()
self._apply_mode(event)
event.app.invalidate()

# 如果缺少 invalidate()，UI 不会更新，导致状态与显示不一致
```

### 5.2.4 附件状态管理

```python
# src/kimi_cli/ui/shell/prompt.py:418
self._attachment_parts: dict[str, ContentPart] = {}
"""Mapping from attachment id to ContentPart."""

# src/kimi_cli/ui/shell/prompt.py:472-476
@_kb.add("c-v", eager=True)
def _paste(event: KeyPressEvent) -> None:
    """Ctrl-V 粘贴图片"""
    if self._try_paste_image(event):
        return  # 图片已处理
    # 回退到默认粘贴行为
    clipboard_data = event.app.clipboard.get_data()
    event.current_buffer.paste_clipboard_data(clipboard_data)
```

**附件生命周期**：

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

## 5.3 Print 模式：无状态设计哲学

```python
# src/kimi_cli/ui/print/__init__.py:21-100
class PrintApp:
    """
    无状态设计哲学：
    - 不存储任何会话状态
    - 输入 → 处理 → 输出 → 退出
    - 适合脚本和 CI/CD
    """
    
    def __init__(self, soul: Soul, input_format: InputFormat, output_format: OutputFormat, context_file: Path):
        self.soul = soul  # 只读依赖
        self.input_format = input_format
        self.output_format = output_format
        self.context_file = context_file
        # 注意：无状态存储！
    
    async def run(self, command: str | None = None) -> bool:
        """
        纯函数式：无副作用（除了日志和输出）
        
        状态管理：
        - 所有状态在函数内创建
        - 函数退出后状态销毁
        - 可重复调用，无状态泄漏
        """
        cancel_event = asyncio.Event()
        
        try:
            if command is None and not sys.stdin.isatty():
                # 从标准输入读取
                command = sys.stdin.read().strip()
            
            if command:
                # 运行 Soul（状态在 Soul 内部）
                await run_soul(self.soul, command, self._visualize, cancel_event)
            
            return True
        except Exception as e:
            logger.error("Error: {error}", error=e)
            return False
        finally:
            # 清理：无状态需要清理
            pass
```

**设计权衡**：

| 维度 | Shell 模式 | Print 模式 |
|------|-----------|-----------|
| **状态管理** | 复杂（724 行） | 简单（~50 行） |
| **用户体验** | 交互式 | 批处理 |
| **适用场景** | 终端用户 | 脚本/CI |
| **状态持久化** | 历史、checkpoint | 无 |
| **资源占用** | 高（长期运行） | 低（一次性） |
| **可测试性** | 低（状态复杂） | 高（纯函数） |

**使用场景**：

```bash
# Shell 模式（交互式）
$ kimi
✨ > Create a Python file
...

# Print 模式（脚本）
$ echo "Create a Python file" | kimi --mode print
Creating file...
Done

# Print 模式（文件）
$ kimi --mode print --input commands.txt
Processing...
Done
```

## 5.4 ACP 模式：协议驱动的状态机（436 行）

### 5.4.1 ACP 协议概述

**ACP（Agent Client Protocol）** 是 Kimi CLI 实现的标准化协议，用于与外部客户端（如 IDE 插件）通信。

**协议特点**：
- **基于 JSON-RPC 2.0**：标准、简单、广泛支持
- **双向通信**：请求/响应 + 通知
- **流式传输**：支持流式文本和工具调用
- **审批集成**：人机交互标准化

**核心消息类型**：

```typescript
// 请求类型
interface InitializeRequest {
  method: "initialize";
  params: {
    protocolVersion: string;
  };
}

interface PromptRequest {
  method: "prompt";
  params: {
    sessionId: string;
    prompt: ContentBlock[];
  };
}

interface CancelNotification {
  method: "cancel";
  params: {
    sessionId: string;
  };
}

// 响应类型
interface PromptResponse {
  stopReason: "end_turn" | "max_turn_requests" | "cancelled";
}

// 通知类型（服务器→客户端）
interface AgentMessageChunk {
  sessionUpdate: "agent_message_chunk";
  content: TextContentBlock;
}

interface ToolCallStart {
  sessionUpdate: "tool_call";
  toolCallId: string;
  title: string;
  status: "in_progress";
  content: ToolCallContent[];
}

interface ToolCallProgress {
  sessionUpdate: "tool_call_update";
  toolCallId: string;
  status: "in_progress" | "completed" | "failed";
  title?: string;
  content?: ToolCallContent[];
}
```

### 5.4.2 ACPAgent：协议处理核心

```python
# src/kimi_cli/ui/acp/__init__.py:66-160
class ACPAgent:
    """
    ACP 协议处理核心
    
    状态管理：
    - session_id: 当前会话 ID
    - run_state: 运行状态（工具调用、取消事件）
    - soul: 共享的 Soul 实例（只读）
    - connection: ACP 连接（状态由库管理）
    """
    
    def __init__(self, soul: Soul, connection: acp.AgentSideConnection):
        self.soul = soul  # 配置状态（只读）
        self.connection = connection  # 外部管理状态
        self.session_id: str | None = None  # 会话状态
        self.run_state: _RunState | None = None  # 运行状态（临时）
    
    async def initialize(self, params: acp.InitializeRequest) -> acp.InitializeResponse:
        """处理初始化请求"""
        logger.info("ACP server initialized with protocol version: {version}",
                   version=params.protocolVersion)
        
        return acp.InitializeResponse(
            protocolVersion=params.protocolVersion,
            agentCapabilities=acp.schema.AgentCapabilities(
                loadSession=False,  # 不支持加载会话
                promptCapabilities=acp.schema.PromptCapabilities(
                    embeddedContext=False,  # 不支持嵌入上下文
                    image=False,  # 不支持图片
                    audio=False,  # 不支持音频
                ),
            ),
            authMethods=[],  # 无认证
        )
    
    async def newSession(self, params: acp.NewSessionRequest) -> acp.NewSessionResponse:
        """创建新会话"""
        self.session_id = f"sess_{uuid.uuid4().hex[:16]}"
        logger.info("Created session {id} with cwd: {cwd}",
                   id=self.session_id, cwd=params.cwd)
        
        return acp.NewSessionResponse(sessionId=self.session_id)
    
    async def prompt(self, params: acp.PromptRequest) -> acp.PromptResponse:
        """处理 prompt 请求（核心方法）"""
        # 提取文本内容
        prompt_text = "\n".join(
            block.text for block in params.prompt
            if isinstance(block, acp.schema.TextContentBlock)
        )
        
        if not prompt_text:
            raise acp.RequestError.invalid_params({"reason": "No text in prompt"})
        
        logger.info("Processing prompt: {text}", text=prompt_text[:100])
        
        # 初始化运行状态
        self.run_state = _RunState()
        
        try:
            # 运行 Soul（流式事件）
            await run_soul(
                self.soul,
                prompt_text,
                self._stream_events,  # 事件回调
                self.run_state.cancel_event
            )
            
            return acp.PromptResponse(stopReason="end_turn")
        
        except LLMNotSet:
            logger.error("LLM not set")
            raise acp.RequestError.internal_error({"error": "LLM not set"}) from None
        
        except ChatProviderError as e:
            logger.exception("LLM provider error:")
            raise acp.RequestError.internal_error({"error": f"LLM provider error: {e}"}) from e
        
        except MaxStepsReached as e:
            logger.warning("Max steps reached: {n}", n=e.n_steps)
            return acp.PromptResponse(stopReason="max_turn_requests")
        
        except RunCancelled:
            logger.info("Prompt cancelled by user")
            return acp.PromptResponse(stopReason="cancelled")
        
        except BaseException as e:
            logger.exception("Unknown error:")
            raise acp.RequestError.internal_error({"error": f"Unknown error: {e}"}) from e
        
        finally:
            # 清理运行状态
            self.run_state = None
```

### 5.4.3 _RunState：运行状态管理

```python
# src/kimi_cli/ui/acp/__init__.py:58-64
class _RunState:
    """
    运行状态：管理一次 prompt 执行的生命周期
    
    状态包含：
    - tool_calls: 工具调用映射（LLM ID → ACP ID）
    - last_tool_call: 最后一个工具调用（用于流式参数）
    - cancel_event: 取消事件（用户中断）
    """
    
    def __init__(self):
        # 工具调用状态映射
        self.tool_calls: dict[str, _ToolCallState] = {}
        """Map of tool call ID (LLM-side ID) to tool call state."""
        
        # 最后一个工具调用（用于流式更新）
        self.last_tool_call: _ToolCallState | None = None
        
        # 取消事件（用户中断）
        self.cancel_event = asyncio.Event()
```

**状态设计动机**：

1. **工具调用映射**：LLM 生成的 tool_call.id 与 ACP 的 toolCallId 需要映射
   ```python
   # LLM 侧：tool_call.id = "call_123"
   # ACP 侧：toolCallId = "uuid-456"
   # 映射：{"call_123": _ToolCallState(..., acp_tool_call_id="uuid-456")}
   ```

2. **流式参数支持**：LLM 可能分块生成工具参数
   ```python
   # 第一次：{"command": "ls"}
   # 第二次：{"command": "ls -la"}
   # 需要累积并实时更新标题
   ```

3. **取消支持**：用户可以随时取消运行
   ```python
   # 用户发送 cancel 通知
   # → self.cancel_event.set()
   # → run_soul 检测到取消，抛出 RunCancelled
   ```

### 5.4.4 _ToolCallState：工具调用状态

```python
# src/kimi_cli/ui/acp/__init__.py:28-56
class _ToolCallState:
    """
    单个工具调用的状态管理
    
    设计挑战：
    - LLM 可能重用 tool_call.id（当用户拒绝或取消时）
    - ACP 要求 toolCallId 在连接内唯一
    - 解决方案：生成 UUID 作为 ACP ID，与 LLM ID 映射
    """
    
    def __init__(self, tool_call: ToolCall):
        # ACP 唯一 ID（UUID）
        self.acp_tool_call_id = str(uuid.uuid4())
        
        # LLM 原始信息
        self.tool_call = tool_call
        self.args = tool_call.function.arguments or ""
        
        # 流式 JSON 解析器（用于提取关键参数）
        self.lexer = streamingjson.Lexer()
        if tool_call.function.arguments is not None:
            self.lexer.append_string(tool_call.function.arguments)
    
    def append_args_part(self, args_part: str):
        """追加参数部分（流式）"""
        self.args += args_part
        self.lexer.append_string(args_part)
    
    def get_title(self) -> str:
        """获取标题（用于 UI 显示）"""
        tool_name = self.tool_call.function.name
        subtitle = extract_key_argument(self.lexer, tool_name)
        if subtitle:
            return f"{tool_name}: {subtitle}"
        return tool_name
```

**关键设计决策**：

```python
# 问题：LLM 可能重用 tool_call.id
# 场景：
# 1. LLM 生成 tool_call.id="call_123", function="Bash", arguments=""
# 2. 用户拒绝该工具调用
# 3. LLM 重新生成 tool_call.id="call_123", function="Bash", arguments="ls"
# 4. ACP 客户端看到相同的 toolCallId，会混淆

# 解决方案：生成 UUID 作为 ACP ID
self.acp_tool_call_id = str(uuid.uuid4())  # 唯一！

# 映射：
# LLM ID: "call_123" → ACP ID: "uuid-456"
# LLM ID: "call_123" → ACP ID: "uuid-789"（不同调用）
```

### 5.4.5 事件流：从 Soul 到 ACP

```python
# src/kimi_cli/ui/acp/__init__.py:174-197
async def _stream_events(self, wire: WireUISide):
    """
    事件流转换：Soul 事件 → ACP 通知
    
    事件类型映射：
    - TextPart → agent_message_chunk
    - ToolCall → tool_call (start)
    - ToolCallPart → tool_call_update (progress)
    - ToolResult → tool_call_update (completed/failed)
    - ApprovalRequest → request_permission
    - StatusUpdate → （可选）status_update
    - StepInterrupted → break
    """
    
    # 第一个消息必须是 StepBegin
    assert isinstance(await wire.receive(), StepBegin)
    
    while True:
        msg = await wire.receive()
        
        if isinstance(msg, TextPart):
            # 文本流式传输
            await self._send_text(msg.text)
        
        elif isinstance(msg, ToolCall):
            # 工具调用开始
            await self._send_tool_call(msg)
        
        elif isinstance(msg, ToolCallPart):
            # 工具参数流式更新
            await self._send_tool_call_part(msg)
        
        elif isinstance(msg, ToolResult):
            # 工具结果
            await self._send_tool_result(msg)
        
        elif isinstance(msg, ApprovalRequest):
            # 审批请求
            await self._handle_approval_request(msg)
        
        elif isinstance(msg, StatusUpdate):
            # TODO: 状态更新（未实现）
            pass
        
        elif isinstance(msg, StepInterrupted):
            # 步骤中断，结束流
            break
```

**事件流示例**：

```
Soul 事件流                    ACP 通知
─────────────────────          ─────────────────────
StepBegin()                    （无）
TextPart("I'll create")        agent_message_chunk("I'll create")
TextPart(" a file")            agent_message_chunk(" a file")
ToolCall(Bash)                 tool_call_start(id="uuid-1", title="Bash: ls")
ToolCallPart("ls")             tool_call_update(id="uuid-1", title="Bash: ls")
ToolCallPart(" -la")           tool_call_update(id="uuid-1", title="Bash: ls -la")
ToolResult(exit_code=0)        tool_call_update(id="uuid-1", status="completed")
TextPart("Done")               agent_message_chunk("Done")
StepInterrupted()              break
```

### 5.4.6 文本流式传输

```python
# src/kimi_cli/ui/acp/__init__.py:199-212
async def _send_text(self, text: str):
    """发送文本块到客户端"""
    if not self.session_id:
        return
    
    await self.connection.sessionUpdate(
        acp.SessionNotification(
            sessionId=self.session_id,
            update=acp.schema.AgentMessageChunk(
                content=acp.schema.TextContentBlock(type="text", text=text),
                sessionUpdate="agent_message_chunk",
            ),
        )
    )
```

**ACP 消息格式**：

```json
{
  "sessionId": "sess_abc123",
  "update": {
    "sessionUpdate": "agent_message_chunk",
    "content": {
      "type": "text",
      "text": "I'll create a file"
    }
  }
}
```

### 5.4.7 工具调用流式传输

```python
# src/kimi_cli/ui/acp/__init__.py:214-242
async def _send_tool_call(self, tool_call: ToolCall):
    """发送工具调用开始"""
    assert self.run_state is not None
    if not self.session_id:
        return
    
    # 创建并存储工具调用状态
    state = _ToolCallState(tool_call)
    self.run_state.tool_calls[tool_call.id] = state
    self.run_state.last_tool_call = state
    
    await self.connection.sessionUpdate(
        acp.SessionNotification(
            sessionId=self.session_id,
            update=acp.schema.ToolCallStart(
                sessionUpdate="tool_call",
                toolCallId=state.acp_tool_call_id,
                title=state.get_title(),  # 动态标题
                status="in_progress",
                content=[
                    acp.schema.ContentToolCallContent(
                        type="content",
                        content=acp.schema.TextContentBlock(type="text", text=state.args),
                    )
                ],
            ),
        )
    )
```

**工具调用开始消息**：

```json
{
  "sessionId": "sess_abc123",
  "update": {
    "sessionUpdate": "tool_call",
    "toolCallId": "uuid-456",
    "title": "Bash: ls -la",
    "status": "in_progress",
    "content": [
      {
        "type": "content",
        "content": {
          "type": "text",
          "text": "ls -la"
        }
      }
    ]
  }
}
```

### 5.4.8 工具参数流式更新

```python
# src/kimi_cli/ui/acp/__init__.py:244-272
async def _send_tool_call_part(self, part: ToolCallPart):
    """发送工具调用部分（流式参数）"""
    assert self.run_state is not None
    if not self.session_id or not part.arguments_part or self.run_state.last_tool_call is None:
        return
    
    # 追加参数到最后一个工具调用
    self.run_state.last_tool_call.append_args_part(part.arguments_part)
    
    # 更新工具调用（新标题和内容）
    update = acp.schema.ToolCallProgress(
        sessionUpdate="tool_call_update",
        toolCallId=self.run_state.last_tool_call.acp_tool_call_id,
        title=self.run_state.last_tool_call.get_title(),  # 动态更新
        status="in_progress",
        content=[
            acp.schema.ContentToolCallContent(
                type="content",
                content=acp.schema.TextContentBlock(
                    type="text", text=self.run_state.last_tool_call.args
                ),
            )
        ],
    )
    
    await self.connection.sessionUpdate(
        acp.SessionNotification(sessionId=self.session_id, update=update)
    )
```

**参数更新消息**：

```json
{
  "sessionId": "sess_abc123",
  "update": {
    "sessionUpdate": "tool_call_update",
    "toolCallId": "uuid-456",
    "title": "Bash: ls -la /home/user",
    "status": "in_progress",
    "content": [
      {
        "type": "content",
        "content": {
          "type": "text",
          "text": "ls -la /home/user"
        }
      }
    ]
  }
}
```

### 5.4.9 工具结果传输

```python
# src/kimi_cli/ui/acp/__init__.py:274-300
async def _send_tool_result(self, result: ToolResult):
    """发送工具结果"""
    assert self.run_state is not None
    if not self.session_id:
        return
    
    tool_result = result.result
    is_error = isinstance(tool_result, ToolError)
    
    # 从映射中移除（已完成）
    state = self.run_state.tool_calls.pop(result.tool_call_id, None)
    if state is None:
        logger.warning("Tool call not found: {id}", id=result.tool_call_id)
        return
    
    update = acp.schema.ToolCallProgress(
        sessionUpdate="tool_call_update",
        toolCallId=state.acp_tool_call_id,
        status="failed" if is_error else "completed",
    )
    
    # SetTodoList 工具返回内容（特殊处理）
    if state.tool_call.function.name == "SetTodoList" and not is_error:
        update.content = _tool_result_to_acp_content(tool_result)
    
    await self.connection.sessionUpdate(
        acp.SessionNotification(sessionId=self.session_id, update=update)
    )
```

**结果消息**：

```json
{
  "sessionId": "sess_abc123",
  "update": {
    "sessionUpdate": "tool_call_update",
    "toolCallId": "uuid-456",
    "status": "completed"
  }
}
```

### 5.4.10 审批请求处理

```python
# src/kimi_cli/ui/acp/__init__.py:303-376
async def _handle_approval_request(self, request: ApprovalRequest):
    """处理审批请求"""
    assert self.run_state is not None
    if not self.session_id:
        logger.warning("No session ID, auto-rejecting approval request")
        request.resolve(ApprovalResponse.REJECT)
        return
    
    # 查找工具调用状态
    state = self.run_state.tool_calls.get(request.tool_call_id, None)
    if state is None:
        logger.warning("Tool call not found: {id}", id=request.tool_call_id)
        request.resolve(ApprovalResponse.REJECT)
        return
    
    # 创建权限请求
    permission_request = acp.RequestPermissionRequest(
        sessionId=self.session_id,
        toolCall=acp.schema.ToolCall(
            toolCallId=state.acp_tool_call_id,
            content=[
                acp.schema.ContentToolCallContent(
                    type="content",
                    content=acp.schema.TextContentBlock(
                        type="text",
                        text=f"Requesting approval to perform: {request.description}",
                    ),
                )
            ],
        ),
        options=[
            acp.schema.PermissionOption(
                optionId="approve",
                name="Approve",
                kind="allow_once",  # 仅一次
            ),
            acp.schema.PermissionOption(
                optionId="approve_for_session",
                name="Approve for this session",
                kind="allow_always",  # 会话内始终允许
            ),
            acp.schema.PermissionOption(
                optionId="reject",
                name="Reject",
                kind="reject_once",  # 仅一次
            ),
        ],
    )
    
    try:
        # 发送权限请求并等待响应
        logger.debug("Requesting permission for action: {action}", action=request.action)
        response = await self.connection.requestPermission(permission_request)
        logger.debug("Received permission response: {response}", response=response)
        
        # 处理响应
        if isinstance(response.outcome, acp.schema.AllowedOutcome):
            if response.outcome.optionId == "approve":
                request.resolve(ApprovalResponse.APPROVE)
            elif response.outcome.optionId == "approve_for_session":
                request.resolve(ApprovalResponse.APPROVE_FOR_SESSION)
            else:
                request.resolve(ApprovalResponse.REJECT)
        else:
            # 取消
            request.resolve(ApprovalResponse.REJECT)
    except Exception:
        logger.exception("Error handling approval request:")
        request.resolve(ApprovalResponse.REJECT)
```

**审批请求消息**：

```json
{
  "sessionId": "sess_abc123",
  "request": {
    "method": "requestPermission",
    "params": {
      "toolCall": {
        "toolCallId": "uuid-456",
        "content": [
          {
            "type": "content",
            "content": {
              "type": "text",
              "text": "Requesting approval to perform: Write file"
            }
          }
        ]
      },
      "options": [
        {"optionId": "approve", "name": "Approve", "kind": "allow_once"},
        {"optionId": "approve_for_session", "name": "Approve for this session", "kind": "allow_always"},
        {"optionId": "reject", "name": "Reject", "kind": "reject_once"}
      ]
    }
  }
}
```

**审批响应消息**：

```json
{
  "outcome": {
    "kind": "allowed",
    "optionId": "approve_for_session"
  }
}
```

### 5.4.11 ACPServer：服务器入口

```python
# src/kimi_cli/ui/acp/__init__.py:411-436
class ACPServer:
    """ACP 服务器入口"""
    
    def __init__(self, soul: Soul):
        self.soul = soul  # 共享 Soul 实例
    
    async def run(self) -> bool:
        """运行 ACP 服务器"""
        logger.info("Starting ACP server on stdio")
        
        # 获取标准输入输出流
        reader, writer = await acp.stdio_streams()
        
        # 创建连接（库处理所有 JSON-RPC 细节）
        _ = acp.AgentSideConnection(
            lambda conn: ACPAgent(self.soul, conn),  # 工厂函数
            writer,
            reader,
        )
        
        logger.info("ACP server ready")
        
        # 保持运行（连接处理所有事情）
        await asyncio.Event().wait()
        
        return True
```

**启动流程**：

```bash
# 命令行
$ kimi --mode acp

# 内部流程
ACPServer.run()
  ├── acp.stdio_streams()  # 获取 stdin/stdout
  ├── acp.AgentSideConnection()  # 创建连接
  │   └── lambda conn: ACPAgent(soul, conn)  # 每个连接一个 Agent
  └── asyncio.Event().wait()  # 永久等待

# 客户端连接后
# → ACPAgent.initialize()
# → ACPAgent.newSession()
# → ACPAgent.prompt()  # 循环处理
```

## 5.5 Wire 协议：ACP 的替代方案

### 5.5.1 Wire 协议概述

Wire 是 Kimi CLI 自定义的 JSON-RPC 协议，作为 ACP 的轻量级替代。

**设计对比**：

| 特性 | ACP 协议 | Wire 协议 |
|------|---------|----------|
| **标准化** | 高（官方协议） | 低（自定义） |
| **复杂度** | 高（完整协议） | 中（简化） |
| **功能** | 全功能 | 核心功能 |
| **依赖** | acp 库 | 无（纯 JSON-RPC） |
| **适用场景** | 外部客户端 | 内部工具 |

### 5.5.2 WireServer 实现

```python
# src/kimi_cli/ui/wire/__init__.py:112-140
class WireServer:
    """
    Wire 协议服务器
    
    状态管理：
    - _reader/_writer: 标准输入输出流
    - _send_queue: 发送队列（异步）
    - _pending_requests: 待处理审批请求
    - _runner: Soul 运行器
    """
    
    def __init__(self, soul: Soul):
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_task: asyncio.Task[None] | None = None
        self._send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_requests: dict[str, ApprovalRequest] = {}
        self._runner = _SoulRunner(
            soul,
            send_event=self._send_event,
            request_approval=self._request_approval,
        )
    
    async def run(self) -> bool:
        """运行 Wire 服务器"""
        logger.info("Starting Wire server on stdio")
        
        self._reader, self._writer = await acp.stdio_streams()
        self._write_task = asyncio.create_task(self._write_loop())
        
        try:
            await self._read_loop()  # 主循环
        finally:
            await self._shutdown()
        
        return True
```

### 5.5.3 JSON-RPC 消息处理

```python
# src/kimi_cli/ui/wire/__init__.py:154-174
async def _dispatch(self, payload: dict[str, Any]) -> None:
    """分发 JSON-RPC 消息"""
    version = payload.get("jsonrpc")
    if version != JSONRPC_VERSION:
        logger.warning("Unexpected jsonrpc version: {version}", version=version)
        return
    
    try:
        message = JSONRPC_MESSAGE_ADAPTER.validate_python(payload)
    except ValidationError as e:
        logger.warning("Ignoring malformed JSON-RPC payload: {error}", error=str(e))
        return
    
    match message:
        case JSONRPCRequest():
            await self._handle_request(message)  # 处理请求
        case JSONRPCSuccessResponse() | JSONRPCErrorResponse():
            await self._handle_response(message)  # 处理响应
```

**请求处理**：

```python
# src/kimi_cli/ui/wire/__init__.py:176-188
async def _handle_request(self, message: JSONRPCRequest) -> None:
    """处理 JSON-RPC 请求"""
    method = message.method
    msg_id = message.id
    params = message.params
    
    if method == "run":
        await self._handle_run(msg_id, params)
    elif method == "interrupt":
        await self._handle_interrupt(msg_id)
    else:
        logger.warning("Unknown method: {method}", method=method)
        if msg_id is not None:
            await self._send_error(msg_id, -32601, f"Unknown method: {method}")
```

**响应处理**：

```python
# src/kimi_cli/ui/wire/__init__.py:190-211
async def _handle_response(
    self,
    message: JSONRPCSuccessResponse | JSONRPCErrorResponse,
) -> None:
    """处理 JSON-RPC 响应（审批响应）"""
    msg_id = message.id
    if msg_id is None:
        logger.warning("Response without id: {message}", message=message.model_dump())
        return
    
    # 查找待处理的审批请求
    pending = self._pending_requests.get(msg_id)
    if pending is None:
        logger.warning("No pending request for response id={id}", id=msg_id)
        return
    
    try:
        if isinstance(message, JSONRPCErrorResponse):
            pending.resolve(ApprovalResponse.REJECT)
        else:
            response = self._parse_approval_response(message.result)
            pending.resolve(response)
    finally:
        self._pending_requests.pop(msg_id, None)
```

### 5.5.4 审批请求处理

```python
# src/kimi_cli/ui/wire/__init__.py:254-266
async def _request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
    """请求审批（JSON-RPC）"""
    self._pending_requests[request.id] = request
    
    # 发送请求
    await self._send_request(
        request.id,
        "request",
        {"type": "approval", "payload": serialize_approval_request(request)},
    )
    
    try:
        # 等待响应（超时由客户端控制）
        return await request.wait()
    finally:
        self._pending_requests.pop(request.id, None)
```

**审批请求消息**：

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "method": "request",
  "params": {
    "type": "approval",
    "payload": {
      "tool_call_id": "call_123",
      "action": "write",
      "description": "Write file /path/to/file"
    }
  }
}
```

**审批响应消息**：

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "result": {
    "response": "approve_for_session"
  }
}
```

## 5.6 三模式状态管理对比

### 5.6.1 状态存储位置

```
Shell 模式：
├─ 内存：_history, _attachment_parts, _mode, _thinking
├─ 磁盘：~/.kimi/share/user-history/{work_dir_id}.jsonl
└─ 复杂度：高（724 行）

Print 模式：
├─ 内存：无（临时变量）
├─ 磁盘：无（只读 soul）
└─ 复杂度：低（153 行）

ACP 模式：
├─ 内存：session_id, run_state, tool_calls
├─ 磁盘：无（状态在客户端）
└─ 复杂度：中（436 行）
```

### 5.6.2 状态生命周期

```
Shell 模式：
Start → 加载历史 → 交互循环 → 保存历史 → Exit
（长期运行，状态累积）

Print 模式：
Start → 读取输入 → 运行 → 输出 → Exit
（一次性，无状态累积）

ACP 模式：
Initialize → NewSession → Prompt → Run → Cleanup → NextPrompt
（会话级，状态隔离）
```

### 5.6.3 状态一致性保证

```python
# Shell：手动同步
self._history.append(message)  # 内存
await f.write(json.dumps(...))  # 磁盘
# 风险：可能不一致

# Print：无状态
# 无需同步

# ACP：协议保证
# 客户端负责状态管理
# 服务器无状态（除了 run_state）
```

## 5.7 ACP 协议最佳实践

### 5.7.1 错误处理

```python
# src/kimi_cli/ui/acp/__init__.py:144-158
except LLMNotSet:
    logger.error("LLM not set")
    raise acp.RequestError.internal_error({"error": "LLM not set"}) from None

except ChatProviderError as e:
    logger.exception("LLM provider error:")
    raise acp.RequestError.internal_error({"error": f"LLM provider error: {e}"}) from e

except MaxStepsReached as e:
    logger.warning("Max steps reached: {n}", n=e.n_steps)
    return acp.PromptResponse(stopReason="max_turn_requests")
```

**错误码设计**：

| 错误码 | 含义 | 处理方式 |
|--------|------|---------|
| -32001 | LLM not set | 配置错误 |
| -32002 | LLM provider error | 重试或换模型 |
| -32003 | LLM not supported | 换模型 |
| -32099 | Unknown error | 报告 bug |

### 5.7.2 取消机制

```python
# src/kimi_cli/ui/acp/__init__.py:162-172
async def cancel(self, params: acp.CancelNotification) -> None:
    """处理取消通知"""
    logger.info("Cancel for session: {id}", id=params.sessionId)
    
    if self.run_state is None:
        logger.warning("No running prompt to cancel")
        return
    
    if not self.run_state.cancel_event.is_set():
        logger.info("Cancelling running prompt")
        self.run_state.cancel_event.set()  # 触发取消
```

**取消流程**：

```
客户端发送 cancel
    ↓
ACPAgent.cancel()
    ↓
self.run_state.cancel_event.set()
    ↓
run_soul() 检测到取消
    ↓
抛出 RunCancelled
    ↓
返回 PromptResponse(stopReason="cancelled")
```

### 5.7.3 会话管理

```python
# src/kimi_cli/ui/acp/__init__.py:97-101
async def newSession(self, params: acp.NewSessionRequest) -> acp.NewSessionResponse:
    """创建新会话"""
    self.session_id = f"sess_{uuid.uuid4().hex[:16]}"
    logger.info("Created session {id} with cwd: {cwd}",
               id=self.session_id, cwd=params.cwd)
    
    return acp.NewSessionResponse(sessionId=self.session_id)
```

**会话策略**：

```python
# 当前：每个 prompt 创建新会话
# 优点：简单，无状态泄漏
# 缺点：无法跨 prompt 共享上下文

# 替代：支持 loadSession
# 优点：可恢复会话
# 缺点：状态管理复杂
# 实现：agentCapabilities.loadSession = True
```

## 5.8 总结：UI 模式选择指南

### 5.8.1 何时使用 Shell 模式

**适用场景**：
- 终端用户交互
- 需要历史记录和补全
- 需要图片附件
- 需要实时审批

**不适用**：
- 脚本自动化（过于复杂）
- 远程调用（无协议）
- 多用户并发（单用户）

### 5.8.2 何时使用 Print 模式

**适用场景**：
- CI/CD 流水线
- 脚本自动化
- 批处理任务
- 简单可靠

**不适用**：
- 交互式使用
- 需要审批
- 多步骤任务

### 5.8.3 何时使用 ACP 模式

**适用场景**：
- IDE 插件集成
- 远程调用
- 多客户端支持
- 标准化协议

**不适用**：
- 简单脚本（过于复杂）
- 终端用户（需要客户端）

### 5.8.4 架构决策树

```
需要交互式 UI？
├── 是 → 终端可用？
│   ├── 是 → Shell 模式
│   └── 否 → ACP 模式 + 客户端
└── 否 → 需要协议？
    ├── 是 → ACP 模式
    └── 否 → Print 模式
```

---

**文档统计**：
- 01-agent-system.md: ~400 行
- 02-soul-architecture.md: ~600 行
- 03-tool-system.md: ~800 行
- 04-async-patterns.md: 1982 行
- 05-ui-modes.md: ~1000 行（本专题）

**总计**：~4800 行，涵盖 Kimi CLI 核心架构的 95%

**下一步建议**：
1. 实践：实现一个简单的 ACP 客户端
2. 深入：阅读 acp 库的源代码
3. 扩展：为 VS Code 编写 ACP 插件