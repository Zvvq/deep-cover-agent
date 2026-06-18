# Deep Cover Agent Runtime

Deep Cover Agent Runtime 是 [Deep Cover](https://github.com/Zvvq/deep-cover) 社交推理游戏的 AI 决策服务。它作为独立的 Python 服务运行，接收 Java 游戏服务端推送的事件，通过 LangChain + DeepSeek 大模型完成 AI 玩家的聊天发言和投票决策。

---

## 项目概述

- **项目定位**：Deep Cover 游戏的 AI Agent 运行时，负责 AI 玩家的智能决策
- **架构角色**：事件驱动的 Python 微服务，与 Java 游戏服务端通过 HTTP 接口双向通信
- **决策引擎**：支持 LangChain + DeepSeek 大模型决策，无 API Key 时自动降级为规则兜底
- **核心能力**：聊天发言决策、投票目标选择、待发送消息复核、空闲主动发言

## 架构设计

```
┌─────────────────────┐     事件推送 (HTTP POST)     ┌──────────────────────┐
│  Java 游戏服务端     │ ───────────────────────────→ │  Python Agent Runtime│
│  (Spring Boot)       │                              │  (FastAPI)            │
│  游戏事实 · 规则校验  │ ←─────────────────────────── │  AI 决策 · 发言生成   │
│  房间状态 · 投票结算  │     内部接口 (HTTP GET/POST)  │  LangChain · DeepSeek│
└─────────────────────┘                              └──────────────────────┘
```

**设计原则**：
- Java 端是游戏状态的唯一事实来源，负责所有规则校验
- Python 端只返回决策建议，Java 端校验后再广播或计票
- Python 不可用时，Java 端自动降级，游戏不受影响

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Web 框架 | FastAPI + Uvicorn |
| AI 框架 | LangChain + langchain-deepseek |
| 大模型 | DeepSeek V4 Flash（支持 OpenAI 兼容中转站） |
| HTTP 客户端 | httpx |
| 数据校验 | Pydantic v2 + pydantic-settings |
| 构建工具 | setuptools (pyproject.toml) |
| 测试框架 | pytest + pytest-asyncio |

## 项目结构

```
src/deep_cover_agent/
├── __init__.py          # 包初始化与版本号
├── __main__.py          # python -m 入口
├── api.py               # FastAPI 应用与 HTTP 接口
├── config.py            # Pydantic Settings 配置管理
├── decision.py          # 决策引擎（LangChain + 规则兜底）
├── java_client.py       # Java 内部接口 HTTP 客户端
├── main.py              # Uvicorn 启动入口
├── models.py            # Pydantic 数据模型
├── runtime.py           # Agent 运行时核心（事件处理、房间记忆、发言调度）
├── tools.py             # LangChain 只读查询工具
└── prompts/
    └── persona_prompt.yml  # AI 人格提示词配置
```

## 核心 API

### 接收游戏事件

```http
POST /agent/rooms/{roomCode}/events
X-Internal-Agent-Secret: dev-agent-secret
Content-Type: application/json
```

Java 服务端向 Agent 推送以下事件类型：

| 事件类型 | 说明 |
|----------|------|
| `ROOM_STARTED` | 游戏开始，携带 AI 玩家列表和话题 |
| `CHAT_MESSAGE` | 新的聊天消息 |
| `ROUND_STARTED` | 新一轮开始 |
| `VOTING_STARTED` | 投票阶段开始 |
| `PLAYER_ELIMINATED` | 玩家被淘汰 |
| `WORD_ROUND_STARTED` | 关键词轮次开始（关键词卧底模式） |
| `WORD_DESCRIPTION_SUBMITTED` | 词语描述已提交（关键词卧底模式） |
| `GAME_ENDED` | 游戏结束 |
| `ROOM_DESTROYED` | 房间被销毁 |

### 健康检查

```http
GET /health
```

### Java 内部接口（Agent 回调）

Agent 通过以下接口向 Java 端查询状态和提交动作：

```http
GET    /api/internal/agent/rooms/{roomCode}/state         # 查询房间状态
GET    /api/internal/agent/rooms/{roomCode}/messages      # 查询最近聊天记录
GET    /api/internal/agent/rooms/{roomCode}/votes         # 查询投票状态
POST   /api/internal/agent/rooms/{roomCode}/messages      # 提交 AI 发言
POST   /api/internal/agent/rooms/{roomCode}/votes         # 提交 AI 投票
```

## 决策引擎

### LangChain + DeepSeek

配置 DeepSeek API Key 后启用，具备：
- **发言决策**：根据聊天上下文、话题和人格提示词，判断是否发言及生成发言内容
- **投票决策**：分析候选人行为和聊天记录，选择投票目标
- **消息复核**：待发送消息等待期间，如果有新真人消息进入，复核剩余草稿是否仍然合适
- **工具调用**：可主动查询 Java 端的房间状态、聊天记录和投票状态

### 规则兜底引擎

无 API Key 或模型调用失败时自动启用：
- **发言决策**：不发言（静默）
- **投票决策**：从合法真人候选人中选择第一个目标
- **消息复核**：继续发送原草稿

## 发言延迟机制

Agent 生成回复后不会立刻提交，而是模拟真人打字延迟：

- 按消息长度计算基础等待时间（`base_delay + length × typing_speed`）
- Agent 可返回 1-3 段短消息，Runtime 逐段提交
- 每段发送前重新查询房间状态，若已不是讨论阶段、AI 已死亡或房间结束则丢弃
- 等待期间有新真人消息时，复核剩余草稿（继续/改写/丢弃/再等一会儿）

可通过 `.env` 调整延迟参数：

```properties
DEEP_COVER_AGENT_SPEECH_BASE_DELAY_SECONDS=1.5
DEEP_COVER_AGENT_SPEECH_TYPING_SECONDS_PER_CHAR=0.5
DEEP_COVER_AGENT_SPEECH_MAX_DELAY_SECONDS=30
DEEP_COVER_AGENT_SPEECH_RETRY_DELAY_SECONDS=3
DEEP_COVER_AGENT_SPEECH_CONTEXT_REACTION_DELAY_SECONDS=2
DEEP_COVER_AGENT_SPEECH_REVISION_EXTRA_DELAY_SECONDS=4
DEEP_COVER_AGENT_SPEECH_MAX_SEGMENTS=3
DEEP_COVER_AGENT_PENDING_SPEECH_MAX_REVIEWS=2
```

## 人格系统

Agent 支持多人格配置，通过 `prompts/persona_prompt.yml` 管理：

- 房间创建时随机选择一个人格，同一房间内固定使用
- 当前内置人格：
  - `casual_blunt` — 嘴直、反应快、有口头禅
  - `lowkey_blunt` — 说话偏短、低调直接
  - `quick_reactor` — 反应快、爱吐槽但不长篇大论
- 修改后需重启 Python Agent 才会生效

## 配置说明

通过 `.env` 文件或环境变量配置（前缀 `DEEP_COVER_AGENT_`）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `JAVA_BASE_URL` | Java 服务端地址 | `http://localhost:8080` |
| `INTERNAL_AGENT_SECRET` | 内部认证密钥 | `dev-agent-secret` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 无（不配置则使用规则兜底） |
| `DEEPSEEK_BASE_URL` | DeepSeek 中转站地址 | 无（使用官方地址） |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `DEEPSEEK_TEMPERATURE` | 生成温度 | `0.7` |
| `DEEPSEEK_THINKING_ENABLED` | 是否启用思考模式 | `false` |
| `ENABLE_LANGCHAIN` | 是否启用 LangChain | `true` |
| `IDLE_CHECK_INTERVAL_SECONDS` | 空闲检查间隔 | `5.0` |
| `IDLE_SPEECH_AFTER_SECONDS` | 空闲多久后主动发言 | `30.0` |
| `MESSAGE_HISTORY_LIMIT` | 消息历史查询上限 | `50` |

> 如果使用 OpenAI 兼容中转站，需设置 `DEEPSEEK_BASE_URL` 为中转站提供的 `/v1` 地址，否则 Key 会被发往默认 DeepSeek 地址并返回 401。
> 默认禁用 DeepSeek thinking mode，避免 LangChain 工具调用循环与 DeepSeek 的 `reasoning_content` 续传要求冲突。

## 快速开始

### 前置条件

- Python 3.11+
- 运行中的 [Deep Cover Java 服务端](https://github.com/Zvvq/deep-cover)

### 安装与启动

```powershell
# 创建并激活 conda 环境
conda create -y -p .conda python=3.11 pip

# 安装项目依赖（含开发工具）
.\.conda\python.exe -m pip install -e ".[dev]"

# 配置环境变量（可选）
# 复制 .env.example 为 .env，设置 DEEP_COVER_AGENT_DEEPSEEK_API_KEY

# 启动服务
.\.conda\python.exe -m uvicorn deep_cover_agent.api:app --host 0.0.0.0 --port 8000 --no-access-log
```

### 验证启动

```http
GET http://localhost:8000/health
```

返回 `{"status": "ok"}` 即表示启动成功。

### 运行测试

```powershell
.\.conda\python.exe -m pytest -q
```

> `pyproject.toml` 中默认禁用了 pytest cacheprovider，避免 Windows 工作区的权限问题。

## Java 端联调配置

在 Java 端的 `application.properties` 中添加：

```properties
deep-cover.agent.enabled=true
deep-cover.agent.base-url=http://localhost:8000
deep-cover.agent.event-path=/agent/rooms/{roomCode}/events
deep-cover.agent.internal-secret=dev-agent-secret
```

## 运行时内部机制

- 按 `eventId` 去重，避免重复处理同一事件
- 按房间维护 AI 玩家列表，忽略 AI 自己发出的聊天事件
- 投票前自动过滤候选人，避免 AI 投给自己
- 房间结束或销毁时自动清理房间记忆

## 许可证

本项目仅供学习使用。
