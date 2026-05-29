from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from .config import Settings
from .decision import (
    DecisionContext,
    DecisionEngine,
    PendingSpeechContext,
    PendingSpeechDecision,
    PersonaPrompt,
    RuleBasedDecisionEngine,
    SpeechDecision,
    select_persona_prompt,
)
from .java_client import JavaAgentClient
from .models import AgentEvent, AgentEventType, Topic


logger = logging.getLogger("uvicorn.error")

GAME_MODE_WORD_UNDERCOVER = "WORD_UNDERCOVER"
STATUS_CHATTING = "CHATTING"
STATUS_DESCRIBING = "DESCRIBING"
STATUS_VOTING = "VOTING"


@dataclass
class PendingSpeechTask:
    ai_player_id: str
    messages: list[str]
    created_after_message_id: str | None
    due_at: float
    next_message_index: int = 0
    sent_messages: list[str] = field(default_factory=list)
    context_changed: bool = False
    review_requested: bool = False
    review_count: int = 0
    processing: bool = False

    @property
    def original_message(self) -> str:
        return "\n".join(self.messages)

    def remaining_messages(self) -> list[str]:
        return self.messages[self.next_message_index :]

    def current_message(self) -> str | None:
        remaining = self.remaining_messages()
        return remaining[0] if remaining else None


@dataclass
class RoomMemory:
    persona: PersonaPrompt = field(default_factory=select_persona_prompt)
    ai_player_ids: set[str] = field(default_factory=set)
    processed_event_ids: set[str] = field(default_factory=set)
    attempted_votes: set[tuple[int, str]] = field(default_factory=set)
    pending_vote_payload: dict[str, Any] | None = None
    pending_speeches: dict[str, PendingSpeechTask] = field(default_factory=dict)
    current_topic: Topic | None = None
    game_mode: str | None = None
    last_activity_at: float = field(default_factory=monotonic)


class WordModeHandler:
    def should_noop(self, memory: RoomMemory, room_state: dict[str, Any] | None = None) -> bool:
        return _is_word_undercover_mode(memory, room_state)

    def log_noop(
        self,
        room_code: str,
        trigger: str,
        memory: RoomMemory,
        room_state: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            "[关键词卧底] room=%s trigger=%s mode=%s status=%s action=noop reason=agent_behavior_not_integrated",
            room_code,
            trigger,
            _room_game_mode(memory, room_state),
            _room_status(room_state),
        )


class AgentRuntime:
    def __init__(
        self,
        java_client: JavaAgentClient,
        decision_engine: DecisionEngine | None = None,
        settings: Settings | None = None,
        persona_selector: Callable[[], PersonaPrompt] | None = None,
        word_mode_handler: WordModeHandler | None = None,
    ) -> None:
        self._java_client = java_client
        self._decision_engine = decision_engine or RuleBasedDecisionEngine()
        self._settings = settings or Settings()
        self._persona_selector = persona_selector or select_persona_prompt
        self._word_mode_handler = word_mode_handler or WordModeHandler()
        self._rooms: dict[str, RoomMemory] = {}
        self._idle_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def room_memory(self, room_code: str) -> RoomMemory:
        memory = self._rooms.get(room_code)
        if memory is None:
            memory = RoomMemory(persona=self._persona_selector())
            self._rooms[room_code] = memory
            logger.info("[房间] room=%s persona=%s selected=true", room_code, memory.persona.name)
        return memory

    def _now(self) -> float:
        return monotonic()

    async def start(self) -> None:
        if self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_loop())
            logger.info(
                "[空闲] loop=started interval=%.1fs threshold=%.1fs",
                self._settings.idle_check_interval_seconds,
                self._settings.idle_speech_after_seconds,
            )

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_task
            self._idle_task = None
            logger.info("[空闲] loop=stopped")
        aclose = getattr(self._java_client, "aclose", None)
        if aclose is not None:
            await aclose()

    async def handle_event(self, room_code: str, event: AgentEvent) -> None:
        memory = self.room_memory(room_code)
        async with self._lock:
            if event.event_id in memory.processed_event_ids:
                logger.debug("[事件] duplicate room=%s type=%s id=%s", room_code, event.type, _short_id(event.event_id))
                return
            memory.processed_event_ids.add(event.event_id)
            memory.last_activity_at = monotonic()

        if event.type == AgentEventType.ROOM_STARTED:
            self._handle_room_started(room_code, memory, event.payload)
        elif event.type == AgentEventType.CHAT_MESSAGE:
            await self._handle_chat_message(room_code, event.payload)
        elif event.type == AgentEventType.VOTING_STARTED:
            await self._handle_voting_started(room_code, event.payload)
        elif event.type == AgentEventType.ROUND_STARTED:
            memory.attempted_votes.clear()
            memory.pending_vote_payload = None
            memory.pending_speeches.clear()
            self._sync_game_mode(memory, event.payload)
            memory.current_topic = _topic_from_payload(event.payload.get("topic"))
            logger.info(
                "[房间] room=%s round=%s topic=%s reset=round_tasks",
                room_code,
                event.payload.get("roundNumber"),
                _topic_text(memory.current_topic),
            )
        elif event.type in {AgentEventType.GAME_ENDED, AgentEventType.ROOM_DESTROYED}:
            self._rooms.pop(room_code, None)
            logger.info("[房间] room=%s closed type=%s", room_code, event.type)

    def _handle_room_started(self, room_code: str, memory: RoomMemory, payload: dict[str, Any]) -> None:
        ai_player_ids = payload.get("aiPlayerIds") or []
        memory.ai_player_ids.update(str(player_id) for player_id in ai_player_ids)
        self._sync_game_mode(memory, payload)
        memory.current_topic = _topic_from_payload(payload.get("topic"))
        logger.info(
            "[房间] room=%s started ai=%s mode=%s topic=%s persona=%s",
            room_code,
            len(memory.ai_player_ids),
            memory.game_mode or "UNKNOWN",
            _topic_text(memory.current_topic),
            memory.persona.name,
        )
        if self._word_mode_handler.should_noop(memory):
            self._word_mode_handler.log_noop(room_code, "room_started", memory)

    async def _handle_chat_message(self, room_code: str, payload: dict[str, Any]) -> None:
        memory = self.room_memory(room_code)
        sender_player_id = str(payload.get("senderPlayerId") or "")
        if sender_player_id in memory.ai_player_ids:
            logger.debug("[聊天] skip=self room=%s ai=%s", room_code, _short_id(sender_player_id))
            return

        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_game_mode(memory, room_state)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if self._word_mode_handler.should_noop(memory, room_state):
                memory.pending_speeches.clear()
                self._word_mode_handler.log_noop(room_code, "chat_message", memory, room_state)
                return
            if room_state.get("status") != STATUS_CHATTING:
                logger.debug("[聊天] skip=status room=%s status=%s", room_code, room_state.get("status"))
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("[聊天][错误] room=%s stage=query error=%s", room_code, _error_summary(exc))
            return

        messages = messages_response.get("messages", [])
        alive_ai_player_ids = self._alive_ai_player_ids(room_state, memory)
        if not alive_ai_player_ids:
            logger.debug("[聊天] skip=no_alive_ai room=%s", room_code)
            return

        logger.info(
            "[聊天] room=%s sender=%s msgs=%s ai=%s",
            room_code,
            _short_id(sender_player_id),
            len(messages),
            len(alive_ai_player_ids),
        )
        for ai_player_id in alive_ai_player_ids:
            if ai_player_id == sender_player_id:
                continue
            pending = memory.pending_speeches.get(ai_player_id)
            if pending is not None:
                pending.context_changed = True
                pending.review_requested = True
                logger.info(
                    "[发言] room=%s ai=%s draft=review_needed sender=%s",
                    room_code,
                    _short_id(ai_player_id),
                    _short_id(sender_player_id),
                )
                continue

            context = DecisionContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages,
                current_topic=memory.current_topic,
                persona_name=memory.persona.name,
                persona_prompt=memory.persona.prompt,
            )
            decision = await self._decision_engine.decide_speech(context)
            speech_parts = _decision_message_parts(decision, self._settings.speech_max_segments)
            if decision.should_speak and speech_parts:
                await self._schedule_speech(room_code, memory, ai_player_id, speech_parts, messages)
            else:
                logger.debug("[发言] room=%s ai=%s decision=silent", room_code, _short_id(ai_player_id))

        await self.run_pending_speech_checks()

    async def _schedule_speech(
        self,
        room_code: str,
        memory: RoomMemory,
        ai_player_id: str,
        speech_parts: list[str],
        conversation_messages: list[dict[str, Any]],
    ) -> None:
        first_part = speech_parts[0]
        delay_seconds = self._speech_delay_seconds(first_part)
        task = PendingSpeechTask(
            ai_player_id=ai_player_id,
            messages=speech_parts[: self._settings.speech_max_segments],
            created_after_message_id=_last_message_id(conversation_messages),
            due_at=self._now() + delay_seconds,
        )
        memory.pending_speeches[ai_player_id] = task
        logger.info(
            "[发言] room=%s ai=%s draft parts=%s delay=%.1fs chars=%s text=%s",
            room_code,
            _short_id(ai_player_id),
            len(task.messages),
            delay_seconds,
            len(first_part),
            _short_text(first_part),
        )

    def _speech_delay_seconds(self, message: str) -> float:
        raw_delay = self._settings.speech_base_delay_seconds + len(message) * self._settings.speech_typing_seconds_per_char
        return max(0.0, min(raw_delay, self._settings.speech_max_delay_seconds))

    async def run_pending_speech_checks(self) -> None:
        now = self._now()
        for room_code, memory in list(self._rooms.items()):
            for ai_player_id, task in list(memory.pending_speeches.items()):
                if task.processing:
                    continue
                if not task.review_requested and task.due_at > now:
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
            logger.debug("[发言] room=%s ai=%s skip=replaced_draft", room_code, _short_id(ai_player_id))
            return
        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_game_mode(memory, room_state)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if self._word_mode_handler.should_noop(memory, room_state):
                memory.pending_speeches.pop(ai_player_id, None)
                self._word_mode_handler.log_noop(room_code, "pending_speech", memory, room_state)
                return
            if room_state.get("status") != STATUS_CHATTING:
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("[发言] room=%s ai=%s discard=status_%s", room_code, _short_id(ai_player_id), room_state.get("status"))
                return
            if ai_player_id not in self._alive_ai_player_ids(room_state, memory):
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("[发言] room=%s ai=%s discard=dead", room_code, _short_id(ai_player_id))
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
        except (httpx.HTTPError, RuntimeError) as exc:
            task.due_at = now + self._settings.speech_retry_delay_seconds
            logger.warning("[发言][错误] room=%s ai=%s stage=pre_send_query retry=%.1fs error=%s", room_code, _short_id(ai_player_id), self._settings.speech_retry_delay_seconds, _error_summary(exc))
            return

        messages = messages_response.get("messages", [])
        new_messages = _human_messages_after(task.created_after_message_id, messages, memory.ai_player_ids)
        context_changed = task.context_changed or bool(new_messages)
        if context_changed:
            review = getattr(self._decision_engine, "review_pending_speech", None)
            if review is None:
                review = RuleBasedDecisionEngine().review_pending_speech
            original_remaining_delay = max(0.0, task.due_at - now)
            context = PendingSpeechContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages,
                original_message=task.original_message,
                sent_messages=list(task.sent_messages),
                remaining_messages=task.remaining_messages(),
                new_messages=new_messages,
                current_topic=memory.current_topic,
                persona_name=memory.persona.name,
                persona_prompt=memory.persona.prompt,
            )
            decision = await review(context)
            task.created_after_message_id = _last_message_id(messages)
            if self._apply_pending_speech_review(room_code, memory, ai_player_id, task, decision, now, original_remaining_delay):
                return
            task.context_changed = False
            task.review_requested = False

        if task.due_at > now:
            return

        message = task.current_message()
        if not message:
            memory.pending_speeches.pop(ai_player_id, None)
            logger.info("[发言] room=%s ai=%s discard=empty", room_code, _short_id(ai_player_id))
            return

        try:
            await self._java_client.send_message(room_code, ai_player_id, message[:300])
        except (httpx.HTTPError, RuntimeError) as exc:
            task.due_at = now + self._settings.speech_retry_delay_seconds
            logger.warning("[发言][错误] room=%s ai=%s stage=submit retry=%.1fs error=%s", room_code, _short_id(ai_player_id), self._settings.speech_retry_delay_seconds, _error_summary(exc))
            return

        task.sent_messages.append(message)
        task.next_message_index += 1
        task.context_changed = False
        task.review_requested = False
        task.review_count = 0
        task.created_after_message_id = _last_message_id(messages)
        if not task.remaining_messages():
            memory.pending_speeches.pop(ai_player_id, None)
            logger.info("[发言] room=%s ai=%s send done=true text=%s", room_code, _short_id(ai_player_id), _short_text(message))
            return

        next_message = task.current_message() or ""
        delay = self._speech_delay_seconds(next_message)
        task.due_at = self._now() + delay
        logger.info("[发言] room=%s ai=%s send next_delay=%.1fs text=%s", room_code, _short_id(ai_player_id), delay, _short_text(message))

    def _apply_pending_speech_review(
        self,
        room_code: str,
        memory: RoomMemory,
        ai_player_id: str,
        task: PendingSpeechTask,
        decision: PendingSpeechDecision,
        now: float,
        original_remaining_delay: float,
    ) -> bool:
        action = decision.action
        if action == "send_original":
            action = "continue"
        elif action == "send_revised":
            action = "revise"

        if action == "wait":
            if task.review_count >= self._settings.pending_speech_max_reviews:
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("[发言] room=%s ai=%s discard=max_reviews", room_code, _short_id(ai_player_id))
                return True
            task.review_count += 1
            delay = decision.extra_delay_seconds if decision.extra_delay_seconds > 0 else self._settings.speech_retry_delay_seconds
            task.due_at = now + min(delay, self._settings.speech_max_delay_seconds)
            task.context_changed = False
            task.review_requested = False
            logger.info("[发言] room=%s ai=%s wait=%.1fs reason=%s", room_code, _short_id(ai_player_id), delay, _short_text(decision.reason))
            return True

        if action == "discard":
            memory.pending_speeches.pop(ai_player_id, None)
            logger.info("[发言] room=%s ai=%s discard=review reason=%s", room_code, _short_id(ai_player_id), _short_text(decision.reason))
            return True

        if action == "revise":
            revised_parts = _pending_decision_message_parts(decision, self._settings.speech_max_segments)
            if not revised_parts:
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("[发言] room=%s ai=%s discard=empty_revision", room_code, _short_id(ai_player_id))
                return True
            task.messages = list(task.sent_messages) + revised_parts
            task.next_message_index = len(task.sent_messages)
            delay = self._revised_speech_delay(revised_parts[0], original_remaining_delay)
            task.due_at = now + delay
            task.context_changed = False
            task.review_requested = False
            logger.info("[发言] room=%s ai=%s revise delay=%.1fs parts=%s reason=%s", room_code, _short_id(ai_player_id), delay, len(revised_parts), _short_text(decision.reason))
            return task.due_at > now

        if action == "continue":
            if not task.remaining_messages():
                memory.pending_speeches.pop(ai_player_id, None)
                logger.info("[发言] room=%s ai=%s discard=no_remaining", room_code, _short_id(ai_player_id))
                return True
            delay = max(original_remaining_delay, self._settings.speech_context_reaction_delay_seconds)
            task.due_at = now + min(delay, self._settings.speech_max_delay_seconds)
            task.context_changed = False
            task.review_requested = False
            logger.info("[发言] room=%s ai=%s continue delay=%.1fs reason=%s", room_code, _short_id(ai_player_id), delay, _short_text(decision.reason))
            return task.due_at > now

        memory.pending_speeches.pop(ai_player_id, None)
        logger.info("[发言] room=%s ai=%s discard=unknown_review_action action=%s", room_code, _short_id(ai_player_id), action)
        return True

    def _revised_speech_delay(self, message: str, original_remaining_delay: float) -> float:
        recomputed_delay = self._speech_delay_seconds(message)
        capped_delay = min(recomputed_delay, original_remaining_delay + self._settings.speech_revision_extra_delay_seconds)
        return max(capped_delay, self._settings.speech_context_reaction_delay_seconds)

    async def _handle_voting_started(self, room_code: str, payload: dict[str, Any]) -> None:
        memory = self.room_memory(room_code)
        memory.pending_speeches.clear()
        self._sync_game_mode(memory, payload)
        if self._word_mode_handler.should_noop(memory, payload):
            memory.pending_vote_payload = None
            self._word_mode_handler.log_noop(room_code, "voting_started", memory, payload)
            return
        memory.pending_vote_payload = dict(payload)
        logger.info("[投票] room=%s started clear_drafts=true", room_code)
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
            self._sync_game_mode(memory, room_state)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if self._word_mode_handler.should_noop(memory, room_state):
                memory.pending_vote_payload = None
                self._word_mode_handler.log_noop(room_code, trigger, memory, room_state)
                return
            if room_state.get("status") != STATUS_VOTING:
                logger.info("[投票] room=%s wait=status_%s trigger=%s", room_code, room_state.get("status"), trigger)
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
            votes_response = await self._java_client.get_votes(room_code)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("[投票][错误] room=%s trigger=%s stage=query error=%s", room_code, trigger, _error_summary(exc))
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
            "[投票] room=%s round=%s ai=%s candidates=%s submitted=%s/%s trigger=%s",
            room_code,
            round_number,
            len(alive_ai_player_ids),
            len(raw_candidates),
            votes_response.get("submittedVoteCount"),
            votes_response.get("requiredVoteCount"),
            trigger,
        )
        for ai_player_id in alive_ai_player_ids:
            vote_key = (round_number, ai_player_id)
            if vote_key in memory.attempted_votes:
                logger.debug("[投票] room=%s round=%s ai=%s skip=duplicate", room_code, round_number, _short_id(ai_player_id))
                continue
            candidates = [
                str(player_id)
                for player_id in raw_candidates
                if str(player_id) != ai_player_id and str(player_id) in alive_player_ids
            ]
            if not candidates:
                logger.info("[投票] room=%s ai=%s skip=no_legal_target", room_code, _short_id(ai_player_id))
                continue

            context = DecisionContext(
                room_code=room_code,
                ai_player_id=ai_player_id,
                room_state=room_state,
                messages=messages_response.get("messages", []),
                current_topic=memory.current_topic,
                persona_name=memory.persona.name,
                persona_prompt=memory.persona.prompt,
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
                logger.info("[投票] room=%s ai=%s decision=wait", room_code, _short_id(ai_player_id))
                continue
            try:
                await self._java_client.cast_vote(room_code, ai_player_id, target, reason)
                memory.attempted_votes.add(vote_key)
                logger.info("[投票] room=%s round=%s ai=%s target=%s reason=%s", room_code, round_number, _short_id(ai_player_id), _short_id(target), _short_text(reason))
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning("[投票][错误] room=%s round=%s ai=%s target=%s stage=submit error=%s", room_code, round_number, _short_id(ai_player_id), _short_id(target), _error_summary(exc))

        if alive_ai_player_ids and all((round_number, ai_player_id) in memory.attempted_votes for ai_player_id in alive_ai_player_ids):
            memory.pending_vote_payload = None
            logger.info("[投票] room=%s round=%s done=true", room_code, round_number)

    def _sync_ai_players(self, memory: RoomMemory, room_state: dict[str, Any]) -> None:
        for player in room_state.get("players", []):
            if player.get("type") == "AI":
                memory.ai_player_ids.add(str(player.get("playerId")))

    def _sync_game_mode(self, memory: RoomMemory, source: dict[str, Any]) -> None:
        if "gameMode" in source and source.get("gameMode") is not None:
            memory.game_mode = str(source.get("gameMode"))

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
            logger.debug("[空闲] room=%s check=true idle=%.1fs", room_code, self._settings.idle_speech_after_seconds)
            await self._idle_speak(room_code)

    async def _idle_speak(self, room_code: str) -> None:
        memory = self.room_memory(room_code)
        try:
            room_state = await self._java_client.get_room_state(room_code)
            self._sync_game_mode(memory, room_state)
            self._sync_ai_players(memory, room_state)
            self._sync_current_topic(memory, room_state)
            if self._word_mode_handler.should_noop(memory, room_state):
                self._word_mode_handler.log_noop(room_code, "idle_speech", memory, room_state)
                return
            if room_state.get("status") != STATUS_CHATTING:
                logger.debug("[空闲] room=%s skip=status_%s", room_code, room_state.get("status"))
                return
            messages_response = await self._java_client.get_messages(room_code, self._settings.message_history_limit)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("[空闲][错误] room=%s stage=query error=%s", room_code, _error_summary(exc))
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
                persona_name=memory.persona.name,
                persona_prompt=memory.persona.prompt,
            )
            decision = await self._decision_engine.decide_speech(context)
            speech_parts = _decision_message_parts(decision, self._settings.speech_max_segments)
            if decision.should_speak and speech_parts:
                await self._schedule_speech(room_code, memory, ai_player_id, speech_parts, messages)
            else:
                logger.debug("[空闲] room=%s ai=%s decision=silent", room_code, _short_id(ai_player_id))
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


def _human_messages_after(message_id: str | None, messages: list[dict[str, Any]], ai_player_ids: set[str]) -> list[dict[str, Any]]:
    return [
        message
        for message in _messages_after(message_id, messages)
        if str(message.get("senderPlayerId") or "") not in ai_player_ids
    ]


def _decision_message_parts(decision: SpeechDecision, max_segments: int) -> list[str]:
    return _normalize_speech_parts(decision.messages, decision.message, max_segments)


def _pending_decision_message_parts(decision: PendingSpeechDecision, max_segments: int) -> list[str]:
    return _normalize_speech_parts(decision.messages, decision.message, max_segments)


def _normalize_speech_parts(parts: tuple[str, ...] | list[str], fallback: str | None, max_segments: int) -> list[str]:
    raw_parts = list(parts) if parts else [fallback]
    normalized: list[str] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, str):
            continue
        part = " ".join(raw_part.split()).strip()
        if not part:
            continue
        normalized.append(part[:300])
        if len(normalized) >= max_segments:
            break
    return normalized


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
        logger.warning("[房间][错误] topic=parse_failed error=%s", _error_summary(exc))
        return None


def _topic_text(topic: Topic | None) -> str:
    if topic is None:
        return "无"
    return topic.content


def _is_word_undercover_mode(memory: RoomMemory, room_state: dict[str, Any] | None = None) -> bool:
    return _room_game_mode(memory, room_state) == GAME_MODE_WORD_UNDERCOVER or _room_status(room_state) == STATUS_DESCRIBING


def _room_game_mode(memory: RoomMemory, room_state: dict[str, Any] | None = None) -> str:
    if room_state is not None and room_state.get("gameMode") is not None:
        return str(room_state.get("gameMode"))
    return memory.game_mode or "UNKNOWN"


def _room_status(room_state: dict[str, Any] | None = None) -> str:
    if room_state is not None and room_state.get("status") is not None:
        return str(room_state.get("status"))
    return "UNKNOWN"


def _short_id(value: str, length: int = 8) -> str:
    return value[:length] if value else "-"


def _error_summary(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if len(message) > 160:
        message = message[:157] + "..."
    return f"{exc.__class__.__name__}: {message}"
