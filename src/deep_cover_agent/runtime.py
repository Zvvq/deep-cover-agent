from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import httpx
from pydantic import ValidationError

from .config import Settings
from .decision import (
    DecisionContext,
    DecisionEngine,
    PendingSpeechContext,
    PendingSpeechDecision,
    RuleBasedDecisionEngine,
)
from .java_client import JavaAgentClient
from .models import AgentEvent, AgentEventType, Topic


logger = logging.getLogger("uvicorn.error")


@dataclass
class PendingSpeechTask:
    ai_player_id: str
    original_message: str
    created_after_message_id: str | None
    due_at: float
    context_changed: bool = False
    review_count: int = 0
    processing: bool = False


@dataclass
class RoomMemory:
    ai_player_ids: set[str] = field(default_factory=set)
    processed_event_ids: set[str] = field(default_factory=set)
    attempted_votes: set[tuple[int, str]] = field(default_factory=set)
    pending_vote_payload: dict[str, Any] | None = None
    pending_speeches: dict[str, PendingSpeechTask] = field(default_factory=dict)
    current_topic: Topic | None = None
    last_activity_at: float = field(default_factory=monotonic)


class AgentRuntime:
    def __init__(
        self,
        java_client: JavaAgentClient,
        decision_engine: DecisionEngine | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._java_client = java_client
        self._decision_engine = decision_engine or RuleBasedDecisionEngine()
        self._settings = settings or Settings()
        self._rooms: dict[str, RoomMemory] = {}
        self._idle_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def room_memory(self, room_code: str) -> RoomMemory:
        return self._rooms.setdefault(room_code, RoomMemory())

    async def start(self) -> None:
        if self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_loop())
            logger.info(
                "空闲检查任务已启动：检查间隔=%s秒，空闲发言阈值=%s秒。",
                self._settings.idle_check_interval_seconds,
                self._settings.idle_speech_after_seconds,
            )

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_task
            self._idle_task = None
            logger.info("空闲检查任务已停止。")
        aclose = getattr(self._java_client, "aclose", None)
        if aclose is not None:
            await aclose()

    async def handle_event(self, room_code: str, event: AgentEvent) -> None:
        memory = self.room_memory(room_code)
        async with self._lock:
            if event.event_id in memory.processed_event_ids:
                logger.info("忽略重复事件：房间=%s，类型=%s，事件ID=%s。", room_code, event.type, event.event_id)
                return
            memory.processed_event_ids.add(event.event_id)
            memory.last_activity_at = monotonic()

        if event.type == AgentEventType.ROOM_STARTED:
            self._handle_room_started(memory, event.payload)
        elif event.type == AgentEventType.CHAT_MESSAGE:
            await self._handle_chat_message(room_code, event.payload)
        elif event.type == AgentEventType.VOTING_STARTED:
            await self._handle_voting_started(room_code, event.payload)
        elif event.type == AgentEventType.ROUND_STARTED:
            memory.attempted_votes.clear()
            memory.pending_vote_payload = None
            memory.pending_speeches.clear()
            memory.current_topic = _topic_from_payload(event.payload.get("topic"))
            logger.info(
                "新回合开始：房间=%s，轮次=%s，当前话题=%s，已清空本轮 AI 临时任务。",
                room_code,
                event.payload.get("roundNumber"),
                _topic_text(memory.current_topic),
            )
        elif event.type in {AgentEventType.GAME_ENDED, AgentEventType.ROOM_DESTROYED}:
            self._rooms.pop(room_code, None)
            logger.info("房间状态已清理：房间=%s，事件类型=%s。", room_code, event.type)

    def _handle_room_started(self, memory: RoomMemory, payload: dict[str, Any]) -> None:
        ai_player_ids = payload.get("aiPlayerIds") or []
        memory.ai_player_ids.update(str(player_id) for player_id in ai_player_ids)
        memory.current_topic = _topic_from_payload(payload.get("topic"))
        logger.info(
            "房间开始：已记录 AI 玩家，数量=%s，AI玩家=%s，当前话题=%s。",
            len(memory.ai_player_ids),
            sorted(memory.ai_player_ids),
            _topic_text(memory.current_topic),
        )

    async def _handle_chat_message(self, room_code: str, payload: dict[str, Any]) -> None:
        memory = self.room_memory(room_code)
        sender_player_id = str(payload.get("senderPlayerId") or "")
        if sender_player_id in memory.ai_player_ids:
            logger.info("忽略 AI 自己的聊天事件：房间=%s，AI玩家=%s。", room_code, sender_player_id)
            return

        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if room_state.get("status") != "CHATTING":
                logger.info("聊天事件暂不处理：房间=%s，当前状态=%s，不是讨论阶段。", room_code, room_state.get("status"))
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("处理聊天事件失败：房间=%s，无法从 Java 查询状态或消息，错误=%s。", room_code, exc)
            return

        messages = messages_response.get("messages", [])
        alive_ai_player_ids = self._alive_ai_player_ids(room_state, memory)
        if not alive_ai_player_ids:
            logger.info("聊天事件无需发言：房间=%s，没有存活的 AI 玩家。", room_code)
            return

        logger.info(
            "开始分析聊天事件：房间=%s，发言玩家=%s，存活AI=%s，消息数=%s。",
            room_code,
            sender_player_id,
            alive_ai_player_ids,
            len(messages),
        )
        for ai_player_id in alive_ai_player_ids:
            if ai_player_id == sender_player_id:
                continue
            pending = memory.pending_speeches.get(ai_player_id)
            if pending is not None:
                pending.context_changed = True
                logger.info(
                    "AI 正在模拟打字，新消息已标记为上下文变化：房间=%s，AI玩家=%s，发言玩家=%s。",
                    room_code,
                    ai_player_id,
                    sender_player_id,
                )
                continue

            context = DecisionContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages,
                current_topic=memory.current_topic,
            )
            decision = await self._decision_engine.decide_speech(context)
            message = (decision.message or "").strip()
            if decision.should_speak and message:
                await self._schedule_speech(room_code, memory, ai_player_id, message[:300], messages)
            else:
                logger.info("AI 决定保持沉默：房间=%s，AI玩家=%s。", room_code, ai_player_id)

        await self.run_pending_speech_checks()

    async def _schedule_speech(
        self,
        room_code: str,
        memory: RoomMemory,
        ai_player_id: str,
        message: str,
        messages: list[dict[str, Any]],
    ) -> None:
        delay_seconds = self._speech_delay_seconds(message)
        task = PendingSpeechTask(
            ai_player_id=ai_player_id,
            original_message=message,
            created_after_message_id=_last_message_id(messages),
            due_at=monotonic() + delay_seconds,
        )
        memory.pending_speeches[ai_player_id] = task
        logger.info(
            "AI 已生成待发送草稿：房间=%s，AI玩家=%s，字数=%s，预计等待=%.2f秒，内容=%s。",
            room_code,
            ai_player_id,
            len(message),
            delay_seconds,
            _short_text(message),
        )

    def _speech_delay_seconds(self, message: str) -> float:
        raw_delay = self._settings.speech_base_delay_seconds + len(message) * self._settings.speech_typing_seconds_per_char
        return max(0.0, min(raw_delay, self._settings.speech_max_delay_seconds))

    async def run_pending_speech_checks(self) -> None:
        now = monotonic()
        for room_code, memory in list(self._rooms.items()):
            for ai_player_id, task in list(memory.pending_speeches.items()):
                if task.processing or task.due_at > now:
                    continue
                task.processing = True
                try:
                    await self._process_pending_speech(room_code, memory, ai_player_id, task, now)
                finally:
                    if memory.pending_speeches.get(ai_player_id) is task:
                        task.processing = False

    async def _process_pending_speech(
        self,
        room_code: str,
        memory: RoomMemory,
        ai_player_id: str,
        task: PendingSpeechTask,
        now: float,
    ) -> None:
        if memory.pending_speeches.get(ai_player_id) is not task:
            logger.info("跳过已被替换的待发送发言：房间=%s，AI玩家=%s。", room_code, ai_player_id)
            return
        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if room_state.get("status") != "CHATTING":
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("丢弃待发送发言：房间=%s，AI玩家=%s，当前状态=%s，不是讨论阶段。", room_code, ai_player_id, room_state.get("status"))
                return
            if ai_player_id not in self._alive_ai_player_ids(room_state, memory):
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("丢弃待发送发言：房间=%s，AI玩家=%s 已不再存活。", room_code, ai_player_id)
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
        except (httpx.HTTPError, RuntimeError) as exc:
            task.due_at = now + self._settings.speech_retry_delay_seconds
            logger.warning("待发送发言复核失败，将稍后重试：房间=%s，AI玩家=%s，错误=%s。", room_code, ai_player_id, exc)
            return

        messages = messages_response.get("messages", [])
        new_messages = _messages_after(task.created_after_message_id, messages)
        context_changed = task.context_changed or bool(new_messages)
        decision = PendingSpeechDecision(action="send_original", message=task.original_message)
        if context_changed:
            review = getattr(self._decision_engine, "review_pending_speech", None)
            if review is None:
                review = RuleBasedDecisionEngine().review_pending_speech
            context = PendingSpeechContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages,
                original_message=task.original_message,
                new_messages=new_messages,
                current_topic=memory.current_topic,
            )
            decision = await review(context)

        action = decision.action
        if action == "wait":
            if task.review_count >= self._settings.pending_speech_max_reviews:
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("丢弃待发送发言：房间=%s，AI玩家=%s，复核等待次数已达到上限。", room_code, ai_player_id)
                return
            task.review_count += 1
            delay = decision.extra_delay_seconds if decision.extra_delay_seconds > 0 else self._settings.speech_retry_delay_seconds
            task.due_at = now + min(delay, self._settings.speech_max_delay_seconds)
            logger.info("AI 决定再等一会儿：房间=%s，AI玩家=%s，等待=%.2f秒，原因=%s。", room_code, ai_player_id, delay, decision.reason)
            return
        if action == "discard":
            memory.pending_speeches.pop(ai_player_id, None)
            logger.info("AI 丢弃过期待发送发言：房间=%s，AI玩家=%s，原因=%s。", room_code, ai_player_id, decision.reason)
            return

        message = task.original_message if action == "send_original" else (decision.message or "").strip()
        if not message:
            memory.pending_speeches.pop(ai_player_id, None)
            logger.info("丢弃待发送发言：房间=%s，AI玩家=%s，复核后消息为空。", room_code, ai_player_id)
            return

        try:
            await self._java_client.send_message(room_code, ai_player_id, message[:300])
            memory.pending_speeches.pop(ai_player_id, None)
            logger.info("AI 已提交发言：房间=%s，AI玩家=%s，动作=%s，内容=%s。", room_code, ai_player_id, action, _short_text(message))
        except (httpx.HTTPError, RuntimeError) as exc:
            task.due_at = now + self._settings.speech_retry_delay_seconds
            logger.warning("AI 发言提交失败，将稍后重试：房间=%s，AI玩家=%s，错误=%s。", room_code, ai_player_id, exc)

    async def _handle_voting_started(self, room_code: str, payload: dict[str, Any]) -> None:
        memory = self.room_memory(room_code)
        memory.pending_speeches.clear()
        memory.pending_vote_payload = dict(payload)
        logger.info("进入投票阶段：房间=%s，已丢弃待发送聊天草稿。", room_code)
        await self._attempt_votes(room_code, memory, memory.pending_vote_payload, "投票开始事件")

    async def _attempt_votes(
        self,
        room_code: str,
        memory: RoomMemory,
        payload: dict[str, Any],
        trigger: str,
    ) -> None:
        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if room_state.get("status") != "VOTING":
                logger.info("投票暂不处理：触发=%s，房间=%s，当前状态=%s，不是投票阶段。", trigger, room_code, room_state.get("status"))
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
            votes_response = await self._java_client.get_votes(room_code)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("处理投票失败：触发=%s，房间=%s，无法从 Java 查询状态、消息或投票信息，错误=%s。", trigger, room_code, exc)
            return

        round_number = int(payload.get("roundNumber") or votes_response.get("roundNumber") or 0)
        raw_candidates = votes_response.get("candidatePlayerIds") or payload.get("candidatePlayerIds") or []
        alive_player_ids = {
            str(player.get("playerId"))
            for player in room_state.get("players", [])
            if player.get("alive") is True and player.get("playerId") is not None
        }

        alive_ai_player_ids = self._alive_ai_player_ids(room_state, memory)
        logger.info(
            "开始分析投票：触发=%s，房间=%s，轮次=%s，存活AI=%s，候选人=%s，已提交=%s/%s。",
            trigger,
            room_code,
            round_number,
            alive_ai_player_ids,
            raw_candidates,
            votes_response.get("submittedVoteCount"),
            votes_response.get("requiredVoteCount"),
        )
        for ai_player_id in alive_ai_player_ids:
            vote_key = (round_number, ai_player_id)
            if vote_key in memory.attempted_votes:
                logger.info("跳过重复投票：房间=%s，轮次=%s，AI玩家=%s。", room_code, round_number, ai_player_id)
                continue
            candidates = [
                str(player_id)
                for player_id in raw_candidates
                if str(player_id) != ai_player_id and str(player_id) in alive_player_ids
            ]
            if not candidates:
                logger.info("AI 无合法投票目标：房间=%s，AI玩家=%s，原始候选人=%s，存活玩家=%s。", room_code, ai_player_id, raw_candidates, sorted(alive_player_ids))
                continue

            context = DecisionContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages_response.get("messages", []),
                current_topic=memory.current_topic,
                candidate_player_ids=candidates,
            )
            decision = await self._decision_engine.decide_vote(context)
            target = decision.target_player_id
            if target not in candidates:
                fallback = await RuleBasedDecisionEngine().decide_vote(context)
                target = fallback.target_player_id
                reason = fallback.reason
            else:
                reason = decision.reason
            if target is None:
                logger.info("AI 决定暂不投票：房间=%s，AI玩家=%s。", room_code, ai_player_id)
                continue
            try:
                await self._java_client.cast_vote(room_code, ai_player_id, target, reason)
                memory.attempted_votes.add(vote_key)
                logger.info("AI 已提交投票：触发=%s，房间=%s，轮次=%s，AI玩家=%s，目标=%s，原因=%s。", trigger, room_code, round_number, ai_player_id, target, reason)
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning("AI 投票提交失败，将等待下一次周期检查重试：触发=%s，房间=%s，轮次=%s，AI玩家=%s，目标=%s，错误=%s。", trigger, room_code, round_number, ai_player_id, target, exc)

        if alive_ai_player_ids and all((round_number, ai_player_id) in memory.attempted_votes for ai_player_id in alive_ai_player_ids):
            memory.pending_vote_payload = None
            logger.info("本轮 AI 投票已处理完成：房间=%s，轮次=%s，AI玩家=%s。", room_code, round_number, alive_ai_player_ids)

    def _sync_ai_players(self, memory: RoomMemory, room_state: dict[str, Any]) -> None:
        for player in room_state.get("players", []):
            if player.get("type") == "AI":
                memory.ai_player_ids.add(str(player.get("playerId")))

    def _sync_current_topic(self, memory: RoomMemory, room_state: dict[str, Any]) -> None:
        if "topic" in room_state:
            memory.current_topic = _topic_from_payload(room_state.get("topic"))

    def _alive_ai_player_ids(self, room_state: dict[str, Any], memory: RoomMemory) -> list[str]:
        player_ids = []
        for player in room_state.get("players", []):
            player_id = str(player.get("playerId"))
            if player_id in memory.ai_player_ids and player.get("alive") is True and player.get("type") == "AI":
                player_ids.append(player_id)
        return player_ids

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.idle_check_interval_seconds)
            await self.run_idle_checks()

    async def run_idle_checks(self) -> None:
        await self.run_pending_speech_checks()
        now = monotonic()
        for room_code, memory in list(self._rooms.items()):
            if memory.pending_vote_payload is not None:
                await self._attempt_votes(room_code, memory, memory.pending_vote_payload, "周期投票检查")
            if memory.pending_speeches:
                continue
            if now - memory.last_activity_at < self._settings.idle_speech_after_seconds:
                continue
            memory.last_activity_at = now
            logger.info("触发空闲发言检查：房间=%s，超过%s秒没有新事件。", room_code, self._settings.idle_speech_after_seconds)
            await self._idle_speak(room_code)

    async def _idle_speak(self, room_code: str) -> None:
        memory = self.room_memory(room_code)
        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if room_state.get("status") != "CHATTING":
                logger.info("空闲检查暂不发言：房间=%s，当前状态=%s，不是讨论阶段。", room_code, room_state.get("status"))
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("空闲检查失败：房间=%s，无法从 Java 查询状态或消息，错误=%s。", room_code, exc)
            return

        messages = messages_response.get("messages", [])
        for ai_player_id in self._alive_ai_player_ids(room_state, memory):
            if ai_player_id in memory.pending_speeches:
                continue
            context = DecisionContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages,
                current_topic=memory.current_topic,
            )
            decision = await self._decision_engine.decide_speech(context)
            message = (decision.message or "").strip()
            if decision.should_speak and message:
                await self._schedule_speech(room_code, memory, ai_player_id, message[:300], messages)
            else:
                logger.info("空闲检查后 AI 保持沉默：房间=%s，AI玩家=%s。", room_code, ai_player_id)
        await self.run_pending_speech_checks()


def _last_message_id(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return None
    message_id = messages[-1].get("messageId")
    return str(message_id) if message_id is not None else None


def _messages_after(message_id: str | None, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if message_id is None:
        return messages
    for index, message in enumerate(messages):
        if str(message.get("messageId")) == message_id:
            return messages[index + 1 :]
    return messages


def _short_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _topic_from_payload(value: Any) -> Topic | None:
    if value is None:
        return None
    try:
        return Topic.model_validate(value)
    except (TypeError, ValueError, ValidationError) as exc:
        logger.warning("忽略无法解析的话题字段：错误=%s，原始值=%s。", exc, value)
        return None


def _topic_text(topic: Topic | None) -> str:
    if topic is None:
        return "无"
    return topic.content
