# 专题 1: Agent 系统与规范加载机制（深度版）

## 1.1 架构哲学：从文本到运行时状态

Agent 系统的核心挑战：**如何将静态的 YAML 配置转化为安全的运行时状态？**

Kimi CLI 采用**三层状态转换**架构：

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 文本层 (YAML)                                      │
│  - 人类可读的配置文件                                       │
│  - 版本控制友好                                             │
│  - 示例：agent.yaml                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ 解析 + 验证
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: 结构化层 (AgentSpec)                              │
│  - Pydantic 模型验证                                        │
│  - 类型安全                                                 │
│  - 可选字段（None 表示未设置）                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ 继承解析 + 路径转换
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: 运行时层 (ResolvedAgentSpec)                      │
│  - 不可变 NamedTuple                                        │
│  - 所有路径转为绝对路径                                     │
│  - 继承链完全解析                                           │
└─────────────────────────────────────────────────────────────┘
```

**状态维度分析**：

| 维度 | 设计决策 | 权衡考量 | 替代方案 |
|------|---------|---------|---------|
| **位置** | `agents/` 目录集中管理 | 优势：易于发现<br>代价：灵活性降低 | 支持任意路径（需安全校验） |
| **生命周期** | 加载时解析，运行时不可变 | 优势：线程安全<br>代价：无法热更新 | 支持热重载（需版本控制） |
| **作用域** | 全局唯一（按名称） | 优势：简单直接<br>代价：命名冲突 | 命名空间隔离 |

## 1.2 核心代码深度剖析

### 1.2.1 AgentSpec：原始配置状态

```python
# src/kimi_cli/agentspec.py:17-30
class AgentSpec(BaseModel):
    """Agent specification - 原始配置状态
    
    设计哲学：
    - 所有字段可为 None，表示"未设置"状态
    - 不施加业务规则，只做结构验证
    - 保持与 YAML 结构的一一对应
    """
    
    extend: str | None = Field(default=None, description="Agent file to extend")
    # 继承指针：相对路径或 "default"
    # None 表示不继承（根 Agent）
    
    name: str | None = Field(default=None, description="Agent name")
    # 最终状态标识：None 表示未设置（将继承父级）
    
    system_prompt_path: Path | None = Field(default=None, description="System prompt path")
    # 系统提示文件路径：相对路径（相对于 agent_file.parent）
    # None 表示未设置（将继承父级）
    
    system_prompt_args: dict[str, str] = Field(default_factory=dict)
    # 系统提示参数：增量合并策略
    # 使用 default_factory 避免可变默认参数陷阱
    # 示例：{"KIMI_WORK_DIR": "/path", "ROLE": "coder"}
    
    tools: list[str] | None = Field(default=None, description="Tools")
    # 工具列表：完整覆盖策略（非合并）
    # None 表示未设置（将继承父级）
    # 示例：["bash", "file", "web"]
    
    exclude_tools: list[str] | None = Field(default=None, description="Tools to exclude")
    # 排除工具：完整覆盖策略
    # 用于从父级中排除特定工具
    
    subagents: dict[str, "SubagentSpec"] | None = Field(default=None, description="Subagents")
    # 子 Agent：完整覆盖策略
    # Key 是子 Agent 名称，Value 是子 Agent 规范
```

**关键设计决策**：

1. **default_factory=dict 而非 default={}**
   ```python
   # 错误示范（可变默认参数陷阱）
   system_prompt_args: dict[str, str] = {}  # 所有实例共享同一个 dict！
   
   # 正确示范
   system_prompt_args: dict[str, str] = Field(default_factory=dict)  # 每个实例独立
   ```

2. **Path 类型而非 str**
   ```python
   # 优势：类型安全，自动验证路径格式
   system_prompt_path: Path | None
   
   # 劣势：需要手动转换相对路径
   agent_spec.system_prompt_path = agent_file.parent / agent_spec.system_prompt_path
   ```

3. **| None 表示可选**
   ```python
   # 明确区分"未设置"和"空值"
   tools: list[str] | None = None  # 未设置（将继承）
   tools: list[str] = []  # 设置为空（不继承，无工具）
   ```

### 1.2.2 继承机制：状态合并的艺术

```python
# src/kimi_cli/agentspec.py:95-114
if agent_spec.extend:
    # 递归加载父配置（深度优先）
    base_agent_spec = _load_agent_spec(base_agent_file)
    
    # 状态合并策略：非简单覆盖，而是智能合并
    
    # 策略 1：直接覆盖（单值字段）
    if agent_spec.name is not None:
        base_agent_spec.name = agent_spec.name
    # 决策逻辑：子类明确指定名称 → 覆盖父类
    
    if agent_spec.system_prompt_path is not None:
        base_agent_spec.system_prompt_path = agent_spec.system_prompt_path
    # 决策逻辑：子类明确指定提示文件 → 覆盖父类
    
    # 策略 2：增量合并（字典字段）
    for k, v in agent_spec.system_prompt_args.items():
        # system prompt args should be merged instead of overwritten
        base_agent_spec.system_prompt_args[k] = v
    # 决策逻辑：参数增量添加/更新，而非完全替换
    # 示例：
    # 父级：{"ROLE": "assistant", "LANG": "en"}
    # 子级：{"LANG": "zh", "MODE": "coding"}
    # 结果：{"ROLE": "assistant", "LANG": "zh", "MODE": "coding"}
    
    # 策略 3：完全覆盖（列表字段）
    if agent_spec.tools is not None:
        base_agent_spec.tools = agent_spec.tools
    # 决策逻辑：工具列表完全替换，不继承父级
    # 权衡：子类精确控制工具集（避免污染）vs 无法增量添加
    # 替代方案：可设计为 tools: list[str] | Literal["inherit"]
    
    if agent_spec.exclude_tools is not None:
        base_agent_spec.exclude_tools = agent_spec.exclude_tools
    
    if agent_spec.subagents is not None:
        base_agent_spec.subagents = agent_spec.subagents
    
    # 状态转移：子类配置完全合并到父类
    agent_spec = base_agent_spec
```

**调用链路分析**：

```
load_agent_spec("custom.yaml")
└── _load_agent_spec("custom.yaml")
    ├── yaml.safe_load() → {"extend": "default", "agent": {"name": "custom"}}
    ├── AgentSpec(**data) → AgentSpec(extend="default", name="custom", ...)
    ├── 路径转换：system_prompt_path = "custom.yaml" 所在目录 + 相对路径
    └── if agent_spec.extend:  # True
        └── _load_agent_spec("default.yaml")  # 递归调用
            ├── yaml.safe_load() → {"agent": {"name": "", "tools": [...]}}
            ├── AgentSpec(**data) → AgentSpec(name="", tools=[...])
            └── if agent_spec.extend:  # False
                └── return AgentSpec(name="", tools=[...])
        # 合并：base_agent_spec.name = "custom"
        # 合并：base_agent_spec.tools = [...]（来自 default）
        └── return AgentSpec(name="custom", tools=[...])
```

**递归深度风险**：
```python
# 潜在问题：循环继承
# agent_a.yaml → extend: agent_b
# agent_b.yaml → extend: agent_a

# 当前实现：无循环检测，会导致无限递归
# 改进方案：
def _load_agent_spec(agent_file: Path, _visited: set[Path] | None = None) -> AgentSpec:
    _visited = _visited or set()
    if agent_file in _visited:
        raise AgentSpecError(f"Circular inheritance detected: {agent_file}")
    _visited.add(agent_file)
    # ... 现有逻辑 ...
```

### 1.2.3 ResolvedAgentSpec：不可变运行时状态

```python
# src/kimi_cli/agentspec.py:40-48
class ResolvedAgentSpec(NamedTuple):
    """Resolved agent specification - 最终不可变状态
    
    设计哲学：
    - 加载后不可变，避免运行时状态漂移
    - 明确区分"配置阶段"和"运行阶段"
    - 类型系统保证完整性（无 None 字段）
    """
    
    name: str  # 不再可为 None，必须有名称
    system_prompt_path: Path  # 绝对路径，解析完成
    system_prompt_args: dict[str, str]  # 合并后的参数字典
    tools: list[str]  # 最终工具列表
    exclude_tools: list[str]  # 可为空列表，但不为 None
    subagents: dict[str, "SubagentSpec"]  # 可为空字典，但不为 None
```

**为什么用 NamedTuple 而非 dataclass？**

```python
# dataclass（可变）
@dataclass
class ResolvedAgentSpec:
    name: str
    # ...

spec = ResolvedAgentSpec(name="test", ...)
spec.name = "modified"  # 允许修改！状态不安全

# NamedTuple（不可变）
class ResolvedAgentSpec(NamedTuple):
    name: str
    # ...

spec = ResolvedAgentSpec(name="test", ...)
spec.name = "modified"  # AttributeError: can't set attribute
```

**不可变性的优势**：

1. **线程安全**：无需锁，可安全共享
2. **可哈希**：可作为 dict key 或放入 set
3. **防御性编程**：防止意外修改
4. **清晰语义**：明确表示"这是最终状态"

**类型转换的代价**：

```python
# 转换过程：AgentSpec → ResolvedAgentSpec
return ResolvedAgentSpec(
    name=agent_spec.name,  # str | None → str（运行时检查）
    system_prompt_path=agent_spec.system_prompt_path,  # Path | None → Path（运行时检查）
    system_prompt_args=agent_spec.system_prompt_args,  # dict（直接传递）
    tools=agent_spec.tools,  # list[str] | None → list[str]（运行时检查）
    exclude_tools=agent_spec.exclude_tools or [],  # None → []
    subagents=agent_spec.subagents or {},  # None → {}
)

# 运行时检查（确保不为 None）
if agent_spec.name is None:
    raise AgentSpecError("Agent name is required")
if agent_spec.system_prompt_path is None:
    raise AgentSpecError("System prompt path is required")
if agent_spec.tools is None:
    raise AgentSpecError("Tools are required")
```

## 1.3 上下文工程实践

### 1.3.1 路径状态管理：相对 vs 绝对

```python
# src/kimi_cli/agentspec.py:90-94
if agent_spec.system_prompt_path is not None:
    # 状态转换：相对路径 → 绝对路径
    # 关键：相对于 agent_file.parent，而非 cwd
    agent_spec.system_prompt_path = agent_file.parent / agent_spec.system_prompt_path
    # 示例：
    # agent_file = "/home/user/project/agents/custom/agent.yaml"
    # system_prompt_path = "./system.md"（相对）
    # 转换后 = "/home/user/project/agents/custom/system.md"（绝对）

if agent_spec.subagents is not None:
    for v in agent_spec.subagents.values():
        # 统一路径基准，避免状态歧义
        v.path = agent_file.parent / v.path
        # 所有子 Agent 路径基于同一基准
```

**路径状态管理的重要性**：

```python
# 错误示范：使用 cwd
agent_spec.system_prompt_path = Path.cwd() / agent_spec.system_prompt_path
# 问题：cwd 可能变化，导致路径失效

# 正确示范：使用 agent_file.parent
agent_spec.system_prompt_path = agent_file.parent / agent_spec.system_prompt_path
# 优势：路径与配置文件位置绑定，稳定可靠
```

**路径状态的生命周期**：

```
YAML 文本（"./system.md"）
    ↓ 解析
AgentSpec.system_prompt_path（Path("./system.md")）
    ↓ 转换（加载时）
AgentSpec.system_prompt_path（Path("/absolute/path/system.md")）
    ↓ 不可变（运行时）
ResolvedAgentSpec.system_prompt_path（Path("/absolute/path/system.md")）
```

### 1.3.2 工具列表状态策略：覆盖 vs 合并

```python
# 关键决策：工具列表是覆盖而非合并
if agent_spec.tools is not None:
    base_agent_spec.tools = agent_spec.tools  # 完全覆盖
```

**设计权衡分析**：

| 策略 | 优势 | 代价 | 适用场景 |
|------|------|------|---------|
| **覆盖**（当前） | 子类精确控制，无工具污染 | 无法增量添加 | 子类需要完全不同的工具集 |
| **合并**（替代） | 可增量添加工具 | 可能引入不需要的工具 | 子类扩展父类功能 |
| **混合**（理想） | 灵活控制 | 复杂性增加 | 所有场景 |

**混合策略实现**：

```python
# 可能的改进方案
tools: list[str] | Literal["inherit"] | None = None
# None = 未设置（继承父级）
# "inherit" = 显式继承（与 None 相同）
# ["tool1", "tool2"] = 完全覆盖

# 或支持增量语法
tools:
  - "+new_tool"  # 添加
  - "-old_tool"  # 移除
  - "=exact_tool"  # 精确设置
```

### 1.3.3 系统提示参数状态：增量合并

```python
# 关键决策：system_prompt_args 是增量合并
for k, v in agent_spec.system_prompt_args.items():
    base_agent_spec.system_prompt_args[k] = v
```

**实际应用场景**：

```yaml
# default/agent.yaml
system_prompt_args:
  ROLE: "general assistant"
  LANG: "en"

# custom/coder.yaml
extend: "default"
system_prompt_args:
  ROLE: "senior software engineer"  # 覆盖
  LANG: "zh"  # 覆盖
  MODE: "coding"  # 新增

# 最终结果：
# ROLE: "senior software engineer"
# LANG: "zh"
# MODE: "coding"
```

**增量合并的优势**：

1. **灵活性**：可覆盖也可新增
2. **可预测性**：明确的行为（更新或添加）
3. **兼容性**：不影响现有参数

**潜在问题**：

```python
# 问题：无法删除父级参数
# 父级：{"UNWANTED": "value"}
# 子级：{}  # 无法删除 UNWANTED

# 解决方案：支持特殊标记
system_prompt_args:
  UNWANTED: "__DELETE__"  # 约定删除标记
```

## 1.4 错误处理与状态恢复

### 1.4.1 验证错误

```python
# src/kimi_cli/agentspec.py:61-66
if agent_spec.name is None:
    raise AgentSpecError("Agent name is required")
if agent_spec.system_prompt_path is None:
    raise AgentSpecError("System prompt path is required")
if agent_spec.tools is None:
    raise AgentSpecError("Tools are required")
```

**错误类型设计**：

```python
# src/kimi_cli/exception.py
class AgentSpecError(Exception):
    """Agent specification error."""
    pass

# 使用场景：
# - 文件不存在 → FileNotFoundError（Python 内置）
# - YAML 格式错误 → yaml.YAMLError → AgentSpecError（包装）
# - 版本不支持 → AgentSpecError
# - 必填字段缺失 → AgentSpecError
```

**错误处理策略**：

```python
try:
    spec = load_agent_spec(agent_file)
except FileNotFoundError:
    # 文件级错误：用户指定的文件不存在
    logger.error(f"Agent file not found: {agent_file}")
    raise
except AgentSpecError as e:
    # 内容级错误：文件存在但内容无效
    logger.error(f"Invalid agent spec: {e}")
    raise
except Exception as e:
    # 未知错误：防御性捕获
    logger.error(f"Unexpected error loading agent: {e}")
    raise AgentSpecError(f"Failed to load agent: {e}") from e
```

### 1.4.2 版本管理

```python
# src/kimi_cli/agentspec.py:85-87
version = data.get("version", 1)
if version != 1:
    raise AgentSpecError(f"Unsupported agent spec version: {version}")
```

**版本策略**：

```python
# 当前：硬编码版本检查
# 未来：支持多版本解析器
_VERSION_PARSERS = {
    1: _parse_v1,
    2: _parse_v2,
}

def _load_agent_spec(agent_file: Path) -> AgentSpec:
    version = data.get("version", 1)
    parser = _VERSION_PARSERS.get(version)
    if parser is None:
        raise AgentSpecError(f"Unsupported version: {version}")
    return parser(data)
```

## 1.5 实际案例：Default Agent 加载流程

```python
# 场景：加载默认 Agent
# 文件：src/kimi_cli/agents/default/agent.yaml

# Step 1: 读取 YAML
version: 1
agent:
  name: ""
  system_prompt_path: ./system.md
  system_prompt_args:
    ROLE_ADDITIONAL: ""
  tools:
    - "kimi_cli.tools.task:Task"
    - "kimi_cli.tools.think:Think"
    # ... 更多工具
  subagents:
    coder:
      path: ./sub.yaml
      description: "Good at general software engineering tasks."

# Step 2: 解析为 AgentSpec
agent_spec = AgentSpec(
    extend=None,  # 不继承
    name="",  # 空字符串（非 None）
    system_prompt_path=Path("./system.md"),  # 相对路径
    system_prompt_args={"ROLE_ADDITIONAL": ""},
    tools=["kimi_cli.tools.task:Task", "kimi_cli.tools.think:Think", ...],
    exclude_tools=None,
    subagents={
        "coder": SubagentSpec(
            path=Path("./sub.yaml"),
            description="Good at general software engineering tasks."
        )
    }
)

# Step 3: 路径转换
agent_spec.system_prompt_path = 
    Path("/Users/jensen/Projects/ai-cli/kimi-cli/src/kimi_cli/agents/default") / "./system.md"
    = Path("/Users/jensen/Projects/ai-cli/kimi-cli/src/kimi_cli/agents/default/system.md")

agent_spec.subagents["coder"].path = 
    Path("/Users/jensen/Projects/ai-cli/kimi-cli/src/kimi_cli/agents/default") / "./sub.yaml"
    = Path("/Users/jensen/Projects/ai-cli/kimi-cli/src/kimi_cli/agents/default/sub.yaml")

# Step 4: 转换为 ResolvedAgentSpec
resolved_spec = ResolvedAgentSpec(
    name="",  # 空字符串（通过验证，因为不是 None）
    system_prompt_path=Path("/Users/jensen/.../agents/default/system.md"),
    system_prompt_args={"ROLE_ADDITIONAL": ""},
    tools=["kimi_cli.tools.task:Task", "kimi_cli.tools.think:Think", ...],
    exclude_tools=[],  # None → []
    subagents={
        "coder": SubagentSpec(
            path=Path("/Users/jensen/.../agents/default/sub.yaml"),
            description="Good at general software engineering tasks."
        )
    }
)
```

**关键观察**：

1. **name="" vs name=None**：空字符串通过验证，None 不通过
2. **extend=None**：表示不继承，递归终止条件
3. **路径转换**：所有相对路径转为绝对路径，避免运行时歧义
4. **空值处理**：None → [] 或 {}，保证下游代码无需检查 None

## 1.6 总结：Agent 系统状态管理精髓

### 1.6.1 状态转换流水线

```
YAML 文本
    ↓ yaml.safe_load()
字典（dict）
    ↓ AgentSpec(**data)
AgentSpec（结构化，含 None）
    ↓ 路径转换 + 继承解析
AgentSpec（结构化，无 None，路径绝对）
    ↓ ResolvedAgentSpec(...)
ResolvedAgentSpec（不可变，运行时安全）
```

### 1.6.2 设计模式应用

1. **Builder 模式**：逐步构建 AgentSpec
2. **Prototype 模式**：通过继承复制和修改
3. **Immutable 模式**：ResolvedAgentSpec 不可变
4. **Null Object 模式**：None → [] / {}（避免空指针）

### 1.6.3 最佳实践

```python
# 1. 使用 default_factory 避免可变默认参数
good: Field(default_factory=dict)
bad: Field(default={})

# 2. 明确区分"未设置"和"空值"
good: list[str] | None = None  # 未设置（将继承）
good: list[str] = []  # 设置为空（不继承）

# 3. 路径转换时机：加载时而非运行时
good: 在 _load_agent_spec 中转换
bad: 在每次使用时转换

# 4. 不可变状态传递
Good: ResolvedAgentSpec(NamedTuple)
Bad: 传递可变对象

# 5. 错误处理：包装底层异常
Good: raise AgentSpecError from e
Bad: 直接抛出底层异常
```

---

**下一步**：专题 2 - Soul 核心架构与上下文管理（将深入 724 行 prompt.py 的复杂状态机）