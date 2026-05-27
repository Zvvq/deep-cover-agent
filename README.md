# Deep Cover Agent Runtime

这是 `deep-cover` 游戏使用的 Python Agent Runtime。Java 端负责游戏事实和规则校验；本服务接收 Java 推送的游戏事件，完成思考和决策，再通过 Java 内部接口提交 AI 发言或投票动作。

## 本地环境

创建并使用项目内的 conda 环境：

```powershell
conda create -y -p .conda python=3.11 pip
.\.conda\python.exe -m pip install -e ".[dev]"
```

复制 `.env.example` 为 `.env`，并设置 `DEEP_COVER_AGENT_DEEPSEEK_API_KEY` 以启用 DeepSeek 决策。没有配置 API Key 时，运行时会使用确定性的兜底决策引擎。

默认会通过 `DEEP_COVER_AGENT_DEEPSEEK_THINKING_ENABLED=false` 禁用 DeepSeek thinking mode，避免 LangChain 工具调用循环与 DeepSeek 的 `reasoning_content` 续传要求冲突。

## 启动

```powershell
.\.conda\python.exe -m uvicorn deep_cover_agent.api:app --host 0.0.0.0 --port 8000 --no-access-log
```

健康检查：

```http
GET http://localhost:8000/health
```

Java 端联调时需要配置：

```properties
deep-cover.agent.enabled=true
deep-cover.agent.base-url=http://localhost:8000
deep-cover.agent.event-path=/agent/rooms/{roomCode}/events
deep-cover.agent.internal-secret=dev-agent-secret
```

## 运行时契约

Java 向 Python 推送事件：

```http
POST /agent/rooms/{roomCode}/events
X-Internal-Agent-Secret: dev-agent-secret
```

Python 调用 Java 内部接口：

```text
GET  /api/internal/agent/rooms/{roomCode}/state
GET  /api/internal/agent/rooms/{roomCode}/messages?limit=50
GET  /api/internal/agent/rooms/{roomCode}/votes
POST /api/internal/agent/rooms/{roomCode}/messages
POST /api/internal/agent/rooms/{roomCode}/votes
```

运行时会按 `eventId` 去重，按房间维护 AI 玩家列表，忽略 AI 自己发出的聊天事件，并在投票前过滤候选人，避免 AI 主动投给自己。

## 测试

```powershell
.\.conda\python.exe -m pytest -q
```

`pyproject.toml` 中默认禁用了 pytest cacheprovider，因为这个 Windows 工作区在早期测试时产生过权限拒绝的 pytest 缓存目录。

## 发言延迟

Agent 生成聊天回复后不会立刻提交给 Java，而是按消息长度模拟打字等待。等待结束后会重新查询 Java 房间状态；如果已经不是讨论阶段、AI 已死亡或房间结束，这条草稿会被丢弃。

如果等待期间有新的真人消息进入，Agent 会在发送前复核旧草稿，决定发送原文、改写后发送、丢弃，或再等待一小段时间。

可通过 `.env` 调整：

```properties
DEEP_COVER_AGENT_SPEECH_BASE_DELAY_SECONDS=2
DEEP_COVER_AGENT_SPEECH_TYPING_SECONDS_PER_CHAR=1
DEEP_COVER_AGENT_SPEECH_MAX_DELAY_SECONDS=45
DEEP_COVER_AGENT_SPEECH_RETRY_DELAY_SECONDS=3
DEEP_COVER_AGENT_PENDING_SPEECH_MAX_REVIEWS=2
```
