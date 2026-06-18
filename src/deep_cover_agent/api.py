from contextlib import asynccontextmanager
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
        "[配置] java=%s langchain=%s deepseek_key=%s thinking=%s idle_check=%.1fs idle_speech=%.1fs",
        settings.java_base_url,
        "on" if settings.enable_langchain else "off",
        "set" if settings.deepseek_api_key is not None else "unset",
        "on" if settings.deepseek_thinking_enabled else "off",
        settings.idle_check_interval_seconds,
        settings.idle_speech_after_seconds,
    )
    java_client = JavaAgentClient(
        base_url=settings.java_base_url,
        internal_secret=settings.internal_agent_secret,
        timeout_seconds=settings.request_timeout_seconds,
    )
    if settings.enable_langchain and settings.deepseek_api_key is not None:
        decision_engine = LangChainDeepSeekDecisionEngine(settings, java_client)
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
        logger.info("[系统] runtime=started")
        try:
            yield
        finally:
            shutdown_runtime = getattr(resolved_runtime, "shutdown", None)
            if shutdown_runtime is not None:
                await shutdown_runtime()
            logger.info("[系统] runtime=stopped")

    app = FastAPI(title="Deep Cover Agent Runtime", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.runtime = resolved_runtime

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = await request.body()
        client = f"{request.client.host}:{request.client.port}" if request.client else "-"
        logger.error(
            "[事件][校验失败] client=%s method=%s path=%s content_type=%s len=%s errors=%s body=%s",
            client,
            request.method,
            request.url.path,
            request.headers.get("content-type"),
            request.headers.get("content-length"),
            exc.errors(),
            body.decode("utf-8", errors="replace")[:300],
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    def require_internal_secret(
        x_internal_agent_secret: Annotated[str | None, Header(alias="X-Internal-Agent-Secret")] = None,
        ) -> None:
        if x_internal_agent_secret != app.state.settings.internal_agent_secret:
            logger.warning("[安全] reject=bad_internal_secret")
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
                "[事件][拒绝] reason=room_mismatch path_room=%s body_room=%s type=%s id=%s",
                room_code,
                event.room_code,
                event.type,
                _short_id(event.event_id),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path room code does not match event room code.",
            )
        logger.info(
            "[事件] room=%s type=%s id=%s %s",
            room_code,
            event.type,
            _short_id(event.event_id),
            _event_summary(event),
        )
        await app.state.runtime.handle_event(room_code, event)
        return {"accepted": True}

    return app


app = create_app()


def _event_summary(event: AgentEvent) -> str:
    payload = event.payload
    if event.type == "CHAT_MESSAGE":
        return "sender=%s text=%s" % (
            _short_id(str(payload.get("senderPlayerId") or "")),
            _short_text(str(payload.get("content") or "")),
        )
    if event.type == "ROOM_STARTED":
        return "ai=%s topic=%s" % (
            len(payload.get("aiPlayerIds") or []),
            _short_text(str((payload.get("topic") or {}).get("content") or "无")),
        )
    if event.type == "ROUND_STARTED":
        return "round=%s topic=%s" % (
            payload.get("roundNumber"),
            _short_text(str((payload.get("topic") or {}).get("content") or "无")),
        )
    if event.type == "VOTING_STARTED":
        return "round=%s candidates=%s" % (
            payload.get("roundNumber"),
            len(payload.get("candidatePlayerIds") or []),
        )
    if event.type == "WORD_ROUND_STARTED":
        return "round=%s current=%s number=%s" % (
            payload.get("roundNumber"),
            _short_id(str(payload.get("currentPlayerId") or "")),
            payload.get("currentNumber"),
        )
    if event.type == "WORD_DESCRIPTION_SUBMITTED":
        description = payload.get("description") or {}
        return "round=%s player=%s number=%s" % (
            payload.get("roundNumber"),
            _short_id(str(description.get("playerId") or "")),
            description.get("number"),
        )
    return ""


def _short_id(value: str, length: int = 8) -> str:
    return value[:length] if value else "-"


def _short_text(text: str, limit: int = 40) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
