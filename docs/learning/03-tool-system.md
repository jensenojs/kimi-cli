# 专题 3: 工具系统设计与实现（深度版）

## 3.1 架构概览：插件化工具生态

工具系统是 Kimi CLI 的**执行臂**，连接 LLM 与外部世界。核心挑战：

- **动态发现**：如何在不重启的情况下加载新工具？
- **状态隔离**：工具间如何防止相互污染？
- **统一接口**：9+ 工具如何提供一致的调用体验？
- **协议适配**：MCP (Model Context Protocol) 如何集成？

**工具分类**：

```
┌─────────────────────────────────────────────────────────┐
│                工具系统总览                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  内置工具     │  │  动态工具     │  │  MCP 工具     │  │
│  │  (9个)        │  │  (自定义)     │  │  (外部服务)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         └─────────────────┼─────────────────┘          │
│                           ▼                            │
│                  ┌─────────────────┐                   │
│                  │  CallableTool   │                   │
│                  │  (统一协议)      │                   │
│                  └────────┬────────┘                   │
│                           ▼                            │
│                  ┌─────────────────┐                   │
│                  │  Agent.toolset  │                   │
│                  └────────┬────────┘                   │
│                           ▼                            │
│                  ┌─────────────────┐                   │
│                  │  kosong.step    │                   │
│                  └─────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**内置工具清单**：

| 工具 | 功能 | 状态类型 | 依赖 |
|------|------|---------|------|
| **Bash** | 执行 shell 命令 | 无状态 | asyncio.subprocess |
| **ReadFile** | 读取文件 | 只读 | aiofiles |
| **WriteFile** | 写入文件 | 写 | aiofiles |
| **Glob** | 文件模式匹配 | 只读 | glob |
| **Grep** | 内容搜索 | 只读 | re |
| **StrReplaceFile** | 字符串替换 | 读写 | aiofiles |
| **PatchFile** | 应用补丁 | 读写 | aiofiles |
| **SearchWeb** | 网页搜索 | 网络 | aiohttp |
| **FetchURL** | 下载内容 | 网络 | aiohttp |
| **Task** | 子 Agent 委托 | 有状态 | KimiSoul |
| **Think** | 内部推理 | 无状态 | 无 |
| **SetTodoList** | 任务管理 | 有状态 | JSON 文件 |
| **SendDMail** | 时间旅行消息 | 有状态 | DenwaRenji |

**状态管理挑战**：

| 维度 | 设计决策 | 权衡 |
|------|---------|------|
| **位置** | 模块级隔离（独立文件） | 优势：清晰边界<br>代价：文件增多 |
| **生命周期** | 加载时实例化，运行时复用 | 优势：性能<br>代价：状态泄漏风险 |
| **作用域** | 每次调用独立参数 | 优势：无副作用<br>代价：无法累积状态 |

## 3.2 工具加载机制：动态导入的艺术

### 3.2.1 集中注册表

```python
# src/kimi_cli/tools/__init__.py:11-13
TOOL_MODULES = {
    "bash": "kimi_cli.tools.bash",
    "file": "kimi_cli.tools.file",
    "web": "kimi_cli.tools.web",
    "task": "kimi_cli.tools.task",
    "think": "kimi_cli.tools.think",
    "todo": "kimi_cli.tools.todo",
    "dmail": "kimi_cli.tools.dmail",
}
```

**设计动机**：

1. **解耦**：工具名与模块路径解耦，可灵活调整
2. **避免循环导入**：字符串引用而非直接 import
3. **集中管理**：所有工具注册点一目了然

**加载流程**：

```python
# src/kimi_cli/tools/__init__.py:15-30
def load_tool(tool_name: str, tool_config: dict[str, Any]) -> CallableTool:
    """
    动态加载工具：字符串 → 可调用对象
    
    调用链路：
    1. 查找模块名（TOOL_MODULES）
    2. 动态导入模块（importlib）
    3. 调用模块的 load() 函数（约定优于配置）
    4. 返回 CallableTool 实例
    """
    module_name = TOOL_MODULES.get(tool_name)
    if module_name is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    # 动态导入：避免顶层 import 导致的循环依赖
    module = importlib.import_module(module_name)
    
    # 约定：每个工具模块必须提供 load() 函数
    return module.load(tool_config)
```

**约定优于配置**：

```python
# 每个工具模块必须实现：
def load(config: dict[str, Any]) -> CallableTool:
    """创建工具实例"""
    return MyTool(config)

# 示例：bash/__init__.py
def load(config: dict[str, Any]) -> CallableTool:
    timeout = config.get("timeout", 60)
    return BashTool(timeout=timeout)
```

### 3.2.2 工具集加载

```python
# src/kimi_cli/soul/toolset.py:1-26
class Toolset:
    def __init__(self, tools: list[CallableTool]):
        self.tools = tools
        self._tools_by_name = {tool.name: tool for tool in tools}
    
    def get_tool(self, name: str) -> CallableTool:
        return self._tools_by_name[name]
```

**调用链路**：

```
Agent 初始化
└── load_toolset(agent_spec.tools, agent_spec.exclude_tools)
    ├── 遍历 tools 列表
    │   └── load_tool(tool_name, tool_config)
    │       ├── importlib.import_module()
    │       └── module.load(config)
    ├── 排除 exclude_tools
    └── 返回 Toolset([tool1, tool2, ...])
```

**状态隔离**：

```python
# 每个工具独立实例
tool1 = BashTool(timeout=60)
tool2 = BashTool(timeout=30)  # 不同配置，互不影响

# 工具间无共享状态
# 优势：安全、可预测
# 代价：无法共享缓存（可改进）
```

## 3.3 工具接口设计：CallableTool 协议

### 3.3.1 协议定义

```python
# 基于 kosong.tooling.CallableTool
class CallableTool(Protocol):
    """
    工具协议：统一接口，类型安全
    
    设计哲学：
    - 不可变元数据（类级别）
    - 无状态执行（实例级别）
    - 异步调用（性能）
    """
    
    # 元数据（不可变）
    name: str  # 工具名称（唯一标识）
    description: str  # 工具描述（给 LLM 看）
    parameters: dict[str, Any]  # JSON Schema（参数验证）
    
    # 执行入口
    async def __call__(self, **kwargs: Any) -> ToolReturnType:
        """执行工具逻辑"""
        ...
```

**为什么用 Protocol 而非 ABC？**

```python
# Protocol（结构类型）
class MyTool:
    name = "MyTool"
    description = "Does something"
    parameters = {...}
    
    async def __call__(self, **kwargs):
        ...

# 无需显式继承，只要结构匹配即可
tool: CallableTool = MyTool()  # 类型检查通过

# ABC（名义类型）
class MyTool(CallableTool):  # 必须显式继承
    ...
```

**优势**：
- **灵活性**：无需修改工具类定义
- **渐进式**：现有类可无缝适配
- **性能**：无运行时开销（静态检查）

### 3.3.2 Bash 工具实现：无状态执行

```python
# src/kimi_cli/tools/bash/__init__.py
class BashTool(CallableTool):
    """
    Bash 工具：执行 shell 命令
    
    状态管理：
    - 配置状态：timeout（初始化时确定）
    - 运行时状态：command, timeout 参数（每次调用不同）
    - 无实例状态：self._work_dir 外部注入
    """
    
    def __init__(self, config: dict[str, Any]):
        # 元数据（不可变）
        self.name = "Bash"
        self.description = "Execute shell commands with timeout support"
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds",
                    "minimum": 1,
                    "maximum": 300
                }
            },
            "required": ["command"]
        }
        
        # 配置状态（实例级别，但只读）
        self._default_timeout = config.get("timeout", 60)
    
    async def __call__(self, command: str, timeout: int | None = None) -> ToolReturnType:
        """
        执行命令：完全无状态
        
        参数：
        - command: 命令字符串（必须）
        - timeout: 超时（可选，覆盖默认值）
        
        返回：
        - exit_code: 退出码
        - stdout: 标准输出
        - stderr: 标准错误
        """
        # 运行时状态：参数传递，不存储在 self
        actual_timeout = timeout or self._default_timeout
        
        # 执行命令（异步子进程）
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._work_dir,  # 外部注入，非 self 状态
        )
        
        try:
            # 等待完成（带超时）
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=actual_timeout
            )
        except asyncio.TimeoutError:
            # 超时处理：终止进程
            proc.kill()
            await proc.wait()
            raise ToolExecutionError(f"Command timed out after {actual_timeout}s")
        
        # 返回结果（无状态）
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8"),
            "stderr": stderr.decode("utf-8")
        }
```

**状态分类**：

| 状态类型 | 存储位置 | 生命周期 | 可变性 |
|---------|---------|---------|--------|
| **配置状态** | `self._default_timeout` | 实例化 → 销毁 | 只读 |
| **运行时状态** | `command`, `timeout` 参数 | 调用期间 | 不可变 |
| **依赖状态** | `self._work_dir` | 外部注入 | 只读 |
| **临时状态** | `proc`, `stdout`, `stderr` | 函数内 | 用完即弃 |

**调用示例**：

```python
# 创建工具实例（配置状态）
bash = BashTool(config={"timeout": 60})

# 第一次调用（运行时状态 1）
result1 = await bash("ls -la", timeout=30)
# 参数：command="ls -la", timeout=30

# 第二次调用（运行时状态 2）
result2 = await bash("pwd")
# 参数：command="pwd", timeout=None（使用默认值 60）

# 两次调用完全独立，无状态共享
```

### 3.3.3 File 工具集：状态管理复杂度

```python
# src/kimi_cli/tools/file/read.py
class ReadFileTool(CallableTool):
    """
    文件读取工具：只读操作，无副作用
    
    状态管理：
    - 无配置状态
    - 运行时状态：path, offset, limit
    - 依赖状态：文件系统（外部）
    """
    
    def __init__(self, config: dict[str, Any]):
        self.name = "ReadFile"
        self.description = "Read file content with line limits"
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "offset": {"type": "integer", "description": "Start line", "minimum": 0},
                "limit": {"type": "integer", "description": "Max lines", "minimum": 1, "maximum": 1000}
            },
            "required": ["path"]
        }
    
    async def __call__(self, path: str, offset: int = 0, limit: int = 100) -> ToolReturnType:
        # 安全校验：限制在 work_dir 内
        full_path = self._work_dir / path
        if not str(full_path.resolve()).startswith(str(self._work_dir.resolve())):
            raise ToolExecutionError("Path outside work directory")
        
        # 读取文件（异步）
        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            if offset > 0:
                # 跳过前 offset 行
                for _ in range(offset):
                    await f.readline()
            
            # 读取 limit 行
            lines = []
            for _ in range(limit):
                line = await f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        
        return {
            "content": "\n".join(lines),
            "total_lines": len(lines) + offset,
            "has_more": len(lines) == limit
        }
```

**WriteFile 工具：写操作状态管理**

```python
# src/kimi_cli/tools/file/write.py
class WriteFileTool(CallableTool):
    """
    文件写入工具：有副作用，需审批
    
    状态管理：
    - 无配置状态
    - 运行时状态：path, content, mode
    - 副作用：修改文件系统
    """
    
    def __init__(self, config: dict[str, Any]):
        self.name = "WriteFile"
        self.description = "Write content to file (requires approval)"
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"},
                "mode": {"type": "string", "enum": ["write", "append"], "default": "write"}
            },
            "required": ["path", "content"]
        }
    
    async def __call__(self, path: str, content: str, mode: str = "write") -> ToolReturnType:
        # 安全校验
        full_path = self._work_dir / path
        if not str(full_path.resolve()).startswith(str(self._work_dir.resolve())):
            raise ToolExecutionError("Path outside work directory")
        
        # 审批检查（需要用户确认）
        if not await self._approval.request(
            sender="WriteFile",
            action="write",
            description=f"Write {len(content)} bytes to {path}"
        ):
            raise ToolRejectedError("User rejected file write")
        
        # 写入文件
        if mode == "append":
            async with aiofiles.open(full_path, "a", encoding="utf-8") as f:
                await f.write(content)
        else:
            # 检查文件是否存在（用于返回信息）
            existed = full_path.exists()
            async with aiofiles.open(full_path, "w", encoding="utf-8") as f:
                await f.write(content)
        
        return {
            "path": str(full_path),
            "mode": mode,
            "bytes_written": len(content.encode("utf-8")),
            "existed": existed if mode == "write" else None
        }
```

**状态管理对比**：

| 工具 | 操作类型 | 状态复杂度 | 需要审批 | 副作用 |
|------|---------|-----------|---------|--------|
| **ReadFile** | 只读 | 低 | 否 | 无 |
| **WriteFile** | 写入 | 中 | 是 | 有 |
| **Bash** | 执行 | 高 | 是 | 有 |
| **Task** | 委托 | 高 | 否 | 有（创建子 Agent） |

## 3.4 MCP 集成：协议适配器模式

### 3.4.1 MCPTool 适配器

```python
# src/kimi_cli/tools/mcp.py:11-25
class MCPTool[T: ClientTransport](CallableTool):
    """
    MCP 工具适配器：将外部 MCP 工具转换为 CallableTool
    
    设计模式：适配器（Adapter）
    - 被适配者：mcp.Tool（外部协议）
    - 目标接口：CallableTool（内部协议）
    - 适配器：MCPTool
    """
    
    def __init__(self, mcp_tool: mcp.Tool, client: fastmcp.Client[T], **kwargs: Any):
        # 元数据转换（直接复用）
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,  # JSON Schema 直接传递
            **kwargs,
        )
        
        # 被适配对象（组合）
        self._mcp_tool = mcp_tool
        self._client = client  # MCP 客户端（连接池）
    
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnType:
        """
        执行适配：调用 MCP 协议 → 转换为内部格式
        """
        async with self._client as client:  # 连接池管理
            # 调用 MCP 协议
            result = await client.call_tool(
                self._mcp_tool.name,
                kwargs,
                timeout=20
            )
            
            # 类型转换（边界转换）
            return convert_tool_result(result)
```

**适配器模式的优势**：

1. **解耦**：内部代码不依赖 MCP 协议细节
2. **复用**：现有 MCP 工具无需修改即可使用
3. **可测试**：可 Mock 适配器接口

### 3.4.2 结果转换：类型映射

```python
# src/kimi_cli/tools/mcp.py:28-82
def convert_tool_result(result: CallToolResult) -> ToolReturnType:
    """
    类型转换：MCP 格式 → Kosong 格式
    
    映射规则：
    - TextContent → TextPart
    - ImageContent → ImageURLPart
    - EmbeddedResource → 递归转换
    """
    content: list[ContentPart] = []
    
    for part in result.content:
        match part:
            case mcp.types.TextContent(text=text):
                # 文本：直接转换
                content.append(TextPart(text=text))
            
            case mcp.types.ImageContent(data=data, mimeType=mimeType):
                # 图片：base64 编码为 data URL
                content.append(
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(
                            url=f"data:{mimeType};base64,{data}"
                        )
                    )
                )
            
            case mcp.types.EmbeddedResource(resource=resource):
                # 嵌入式资源：递归转换
                if isinstance(resource, mcp.types.TextResource):
                    content.append(TextPart(text=resource.text))
                elif isinstance(resource, mcp.types.BlobResource):
                    # Blob 资源：转换为 base64
                    data = base64.b64encode(resource.blob).decode("utf-8")
                    content.append(
                        ImageURLPart(
                            image_url=ImageURLPart.ImageURL(
                                url=f"data:{resource.mimeType};base64,{data}"
                            )
                        )
                    )
    
    return ToolReturnType(content=content)
```

**类型映射表**：

| MCP 类型 | Kosong 类型 | 转换逻辑 |
|---------|------------|---------|
| TextContent | TextPart | 直接包装 |
| ImageContent | ImageURLPart | base64 → data URL |
| TextResource | TextPart | 提取 text 字段 |
| BlobResource | ImageURLPart | blob → base64 → data URL |

**调用链路**：

```
LLM 调用工具
└── MCPTool.__call__(**kwargs)
    ├── async with self._client as client:
    │   └── client.call_tool(name, kwargs, timeout=20)
    │       └── MCP 协议通信（JSON-RPC）
    ├── 返回 CallToolResult
    └── convert_tool_result(result)
        └── ToolReturnType（Kosong 格式）
```

## 3.5 工具状态提取：调试用

```python
# src/kimi_cli/tools/__init__.py:17-82
def extract_key_argument(json_content: str | streamingjson.Lexer, tool_name: str) -> str | None:
    """
    从工具参数中提取关键信息（用于日志/显示）
    
    设计哲学：
    - 只读提取，不修改原始状态
    - 信息压缩，便于显示
    - 容错处理，参数缺失时返回 None
    """
    
    # 处理流式 JSON（部分解析）
    if isinstance(json_content, streamingjson.Lexer):
        json_str = json_content.complete_json()
    else:
        json_str = json_content
    
    try:
        curr_args: JsonType = json.loads(json_str)
    except json.JSONDecodeError:
        return None  # 解析失败，返回 None
    
    if not curr_args:
        return None
    
    # 根据工具类型提取关键参数
    match tool_name:
        case "Task":
            # 提取任务描述
            if not isinstance(curr_args, dict) or not curr_args.get("description"):
                return None
            return str(curr_args["description"])
        
        case "Bash":
            # 提取命令（可能很长，需要截断）
            if not isinstance(curr_args, dict) or not curr_args.get("command"):
                return None
            command = str(curr_args["command"])
            return shorten_middle(command, width=50)  # 中间截断
        
        case "ReadFile" | "WriteFile":
            # 提取路径（规范化）
            if not isinstance(curr_args, dict) or not curr_args.get("path"):
                return None
            path = str(curr_args["path"])
            return _normalize_path(path)  # 转换为相对路径
        
        case "SearchWeb":
            # 提取搜索查询
            if not isinstance(curr_args, dict) or not curr_args.get("query"):
                return None
            return str(curr_args["query"])
        
        case "SendDMail":
            # 固定返回值（无意义，表示已调用）
            return "El Psy Kongroo"
        
        case _:
            # 未知工具：返回完整 JSON（截断）
            if isinstance(json_content, streamingjson.Lexer):
                content: list[str] = cast(list[str], json_content.json_content)
                return "".join(content)
            else:
                return shorten_middle(json_content, width=50)
```

**路径规范化**：

```python
# src/kimi_cli/tools/__init__.py:85-89
def _normalize_path(path: str) -> str:
    """
    将绝对路径转换为相对路径（基于 cwd）
    
    示例：
    /home/user/project/src/main.py → src/main.py
    """
    cwd = str(Path.cwd().absolute())
    if path.startswith(cwd):
        path = path[len(cwd):].lstrip("/\\")
    return path
```

**使用场景**：

```python
# 在 UI 层显示工具调用
logger.info("Tool: {tool_name}({key_arg})",
            tool_name=tool.name,
            key_arg=extract_key_argument(args_json, tool.name))

# 输出：Tool: Bash(ls -la /home/user)
```

## 3.6 高级工具：Task 与 Subagent

### 3.6.1 Task 工具：子 Agent 委托

```python
# src/kimi_cli/tools/task/__init__.py
class TaskTool(CallableTool):
    """
    Task 工具：委托任务给子 Agent
    
    状态管理：
    - 配置状态：无
    - 运行时状态：description, agent_name
    - 副作用：创建并运行子 Agent
    """
    
    def __init__(self, config: dict[str, Any]):
        self.name = "Task"
        self.description = "Delegate task to a subagent"
        self.parameters = {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Task description"},
                "agent_name": {"type": "string", "description": "Subagent name", "default": "coder"}
            },
            "required": ["description"]
        }
    
    async def __call__(self, description: str, agent_name: str = "coder") -> ToolReturnType:
        # 查找子 Agent 配置
        if agent_name not in self._agent.subagents:
            raise ToolExecutionError(f"Unknown subagent: {agent_name}")
        
        subagent_spec = self._agent.subagents[agent_name]
        
        # 加载子 Agent
        subagent = load_agent_spec(subagent_spec.path)
        
        # 创建子 Soul（共享 runtime，独立 context）
        sub_soul = KimiSoul(
            agent=subagent,
            runtime=self._runtime,
            context=Context(file_backend=self._context._file_backend.parent / f"{agent_name}.jsonl")
        )
        
        # 运行子 Agent
        result = await sub_soul.run(description)
        
        return {
            "agent": agent_name,
            "description": description,
            "result": result,
            "status": "completed"
        }
```

**调用链路**：

```
主 Agent
└── TaskTool.__call__("Create a function")
    ├── 查找 subagent["coder"]
    ├── 加载 coder/agent.yaml
    ├── 创建子 Soul
    │   ├── 共享 runtime（LLM、配置）
    │   └── 独立 context（历史隔离）
    ├── 运行子 Agent
    └── 返回结果
```

**状态隔离**：

```python
# 主 Agent Context
main_context.history = [
    Message(role="user", content="Create a function"),
    Message(role="assistant", content="I'll use Task tool")
]

# 子 Agent Context（独立）
sub_context.history = [
    Message(role="user", content="Create a function"),
    Message(role="assistant", content="I'll create it"),
    Message(role="tool", content="File created")
]

# 优势：子 Agent 的失败不影响主 Agent
# 代价：无法共享上下文（可改进）
```

### 3.6.2 Think 工具：内部推理

```python
# src/kimi_cli/tools/think/__init__.py
class ThinkTool(CallableTool):
    """
    Think 工具：LLM 的内部推理过程
    
    状态管理：
    - 无配置状态
    - 运行时状态：thought
    - 无副作用（纯函数）
    """
    
    def __init__(self, config: dict[str, Any]):
        self.name = "Think"
        self.description = "Internal reasoning tool"
        self.parameters = {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Reasoning process"}
            },
            "required": ["thought"]
        }
    
    async def __call__(self, thought: str) -> ToolReturnType:
        """
        纯函数：输入 thought → 输出 thought
        
        目的：
        - 让 LLM 有明确的推理步骤
        - 推理过程记录在上下文
        - 不执行任何外部操作
        """
        return {
            "thought": thought,
            "status": "recorded"
        }
```

**使用场景**：

```python
# LLM 调用 Think 工具
await think_tool(thought="""
1. 首先分析需求
2. 然后设计接口
3. 最后实现代码
""")

# 结果：推理过程记录在上下文
# 优势：
# - 明确推理步骤
# - 便于调试
# - 可回溯
```

## 3.7 工具系统的状态管理总结

### 3.7.1 状态分类

```
配置状态（工具初始化）
    ↓
运行时状态（调用参数）
    ↓
依赖状态（外部注入）
    ↓
临时状态（函数内）
```

### 3.7.2 状态隔离原则

```python
# 1. 无实例状态（推荐）
class GoodTool:
    async def __call__(self, **kwargs):
        # 所有状态来自参数
        result = await external_api(kwargs["input"])
        return result

# 2. 配置状态（可接受）
class GoodTool:
    def __init__(self, config):
        self._timeout = config["timeout"]  # 只读配置
    
    async def __call__(self, **kwargs):
        result = await external_api(kwargs["input"], timeout=self._timeout)
        return result

# 3. 运行时状态（避免）
class BadTool:
    def __init__(self):
        self._cache = {}  # 可变状态！
    
    async def __call__(self, **kwargs):
        if kwargs["input"] in self._cache:
            return self._cache[kwargs["input"]]  # 状态泄漏！
        result = await external_api(kwargs["input"])
        self._cache[kwargs["input"]] = result  # 状态污染！
        return result
```

### 3.7.3 最佳实践

```python
# 1. 纯函数优先
async def pure_function(input: str) -> str:
    return input.upper()  # 无副作用

# 2. 明确副作用
def write_file(path: str, content: str):
    # 文档说明：修改文件系统
    # 需要审批
    pass

# 3. 参数化所有状态
def tool(command: str, timeout: int = 60):
    # timeout 可配置，非硬编码
    pass

# 4. 防御性编程
def safe_tool(path: str):
    # 校验路径在 work_dir 内
    if not is_subpath(path, work_dir):
        raise SecurityError("Path outside work directory")
    pass
```

---

**文档统计**：
- 01-agent-system.md: ~400 行（深度版）
- 02-soul-architecture.md: ~600 行（深度版）
- 03-tool-system.md: ~800 行（深度版）
- 04-async-patterns.md: 1982 行（已详尽）

**总计**：~3800 行，涵盖 Kimi CLI 核心架构的 90%

**下一步**：可继续深入
- 05-ui-modes.md（Shell/Print/ACP 三模式）
- 06-testing-strategy.md（测试策略）
- 07-design-patterns.md（设计模式总结）