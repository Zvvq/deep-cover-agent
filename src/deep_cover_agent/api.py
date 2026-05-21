from contextlib import asynccontextmanager
import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .decision import LangChainDeepSeekDecisionEngine, RuleBasedDecisionEngine
from .java_client import JavaAgentClient
from .models import AgentEvent
from .runtime import AgentRuntime


logger = logging.getLogger("uvicorn.error")


def build_runtime(settings: Settings) -> AgentRuntime:
    logger.info(
        "Agent Runtime 配置：Java地址=%s，LangChain=%s，DeepSeek密钥=%s，空闲检查=%s秒，空闲发言阈值=%s秒",
        settings.java_base_url,
        "启用" if settings.enable_langchain else "关闭",
        "已配置" if settings.deepseek_api_key is not None else "未配置",
        settings.idle_check_interval_seconds,
        settings.idle_speech_after_seconds,
    )
    java_client = JavaAgentClient(
        base_url=settings.java_base_url,
        internal_secret=settings.internal_agent_secret,
        timeout_seconds=settings.request_timeout_seconds,
    )
    if settings.enable_langchain and settings.deepseek_api_key is not None:
        decision_engine = LangChainDeepSeekDecisionEngine(settings)
    else:
        decision_engine = RuleBasedDecisionEngine()
    return AgentRuntime(java_client, decision_engine, settings)


def create_app(settings: Settings | None = None, runtime: AgentRuntime | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_runtime = runtime or build_runtime(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        start = getattr(resolved_runtime, "start", None)
        if start is not None:
            await start()
        logger.info("Agent Runtime 已启动，等待 Java 推送游戏事件。")
        try:
            yield
        finally:
            shutdown_runtime = getattr(resolved_runtime, "shutdown", None)
            if shutdown_runtime is not None:
                await shutdown_runtime()
            logger.info("Agent Runtime 已停止。")

    app = FastAPI(title="Deep Cover Agent Runtime", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.runtime = resolved_runtime

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = await request.body()
        client = f"{request.client.host}:{request.client.port}" if request.client else "-"
        logger.error(
            "Agent 事件校验失败：客户端=%s，方法=%s，路径=%s，内容类型=%s，内容长度=%s，"
            "User-Agent=%s，错误=%s，请求体=%s",
            client,
            request.method,
            request.url.path,
            request.headers.get("content-type"),
            request.headers.get("content-length"),
            request.headers.get("user-agent"),
            exc.errors(),
            body.decode("utf-8", errors="replace")[:4000],
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    def require_internal_secret(
        x_internal_agent_secret: Annotated[str | None, Header(alias="X-Internal-Agent-Secret")] = None,
        ) -> None:
        if x_internal_agent_secret != app.state.settings.internal_agent_secret:
            logger.warning("拒绝内部请求：X-Internal-Agent-Secret 不正确或缺失。")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid internal agent secret.",
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/agent/rooms/{room_code}/events", status_code=status.HTTP_202_ACCEPTED)
    async def receive_event(
        room_code: str,
        event: AgentEvent,
        _: None = Depends(require_internal_secret),
    ) -> dict[str, bool]:
        if event.room_code != room_code:
            logger.warning(
                "拒绝 Agent 事件：路径房间=%s 与事件房间=%s 不一致，事件ID=%s，类型=%s。",
                room_code,
                event.room_code,
                event.event_id,
                event.type,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path room code does not match event room code.",
            )
        logger.info(
            "收到 Java 事件：房间=%s，类型=%s，事件ID=%s，创建时间=%s，payload=%s",
            room_code,
            event.type,
            event.event_id,
            event.created_at.isoformat(),
            _compact_json(event.payload),
        )
        await app.state.runtime.handle_event(room_code, event)
        logger.info("事件处理完成：房间=%s，类型=%s，事件ID=%s。", room_code, event.type, event.event_id)
        return {"accepted": True}

    return app


app = create_app()


def _compact_json(value: object, limit: int = 800) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
