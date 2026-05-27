import asyncio

import pytest

from deep_cover_agent.config import Settings
from deep_cover_agent.decision import PendingSpeechDecision, SpeechDecision, VoteDecision
from deep_cover_agent.models import AgentEvent, AgentEventType, Topic
from deep_cover_agent.runtime import AgentRuntime


class FakeJavaClient:
    def __init__(self) -> None:
        self.room_state = {
            "roomCode": "ABC123",
            "status": "CHATTING",
            "roundNumber": 1,
            "aliveHumanCount": 2,
            "aliveAiCount": 1,
            "players": [
                {"playerId": "human-1", "type": "HUMAN", "alive": True, "host": True},
                {"playerId": "human-2", "type": "HUMAN", "alive": True, "host": False},
                {"playerId": "ai-1", "type": "AI", "alive": True, "host": False},
            ],
        }
        self.messages = {
            "roomCode": "ABC123",
            "messages": [
                {
                    "messageId": "message-1",
                    "senderPlayerId": "human-1",
                    "content": "I went hiking last weekend.",
                    "createdAt": "2026-05-18T01:00:00Z",
                }
            ],
        }
        self.vote_state = {
            "roomCode": "ABC123",
            "roundNumber": 1,
            "submittedVoteCount": 0,
            "requiredVoteCount": 3,
            "candidatePlayerIds": ["human-1", "human-2", "ai-1"],
        }
        self.sent_messages = []
        self.cast_votes = []
        self.fail_vote_attempts = 0

    async def get_room_state(self, room_code: str):
        return self.room_state

    async def get_messages(self, room_code: str, limit: int = 50):
        return self.messages

    async def get_votes(self, room_code: str):
        return self.vote_state

    async def send_message(self, room_code: str, ai_player_id: str, content: str):
        self.sent_messages.append((room_code, ai_player_id, content))
        return {"id": "message-ai", "roomCode": room_code, "senderPlayerId": ai_player_id, "content": content}

    async def cast_vote(self, room_code: str, ai_player_id: str, target_player_id: str, reason: str):
        if self.fail_vote_attempts > 0:
            self.fail_vote_attempts -= 1
            raise RuntimeError("temporary vote submit failure")
        self.cast_votes.append((room_code, ai_player_id, target_player_id, reason))
        return {"roomCode": room_code, "roundNumber": 1, "settled": False}


class SlowSendJavaClient(FakeJavaClient):
    async def send_message(self, room_code: str, ai_player_id: str, content: str):
        await asyncio.sleep(0.01)
        return await super().send_message(room_code, ai_player_id, content)


class ScriptedDecisionEngine:
    def __init__(self) -> None:
        self.speech_calls = []
        self.vote_calls = []
        self.pending_speech_reviews = []
        self.pending_speech_decision = PendingSpeechDecision(action="send_original")

    async def decide_speech(self, context):
        self.speech_calls.append(context)
        return SpeechDecision(should_speak=True, message="That sounds familiar to me.")

    async def decide_vote(self, context):
        self.vote_calls.append(context)
        return VoteDecision(target_player_id=context.candidate_player_ids[0], reason="Most suspicious.")

    async def review_pending_speech(self, context):
        self.pending_speech_reviews.append(context)
        return self.pending_speech_decision


def immediate_settings() -> Settings:
    return Settings(
        speech_base_delay_seconds=0,
        speech_typing_seconds_per_char=0,
        speech_max_delay_seconds=0,
        speech_retry_delay_seconds=0,
    )


def event(event_id: str, event_type: AgentEventType, payload: dict) -> AgentEvent:
    return AgentEvent.model_validate(
        {
            "eventId": event_id,
            "type": event_type,
            "roomCode": "ABC123",
            "createdAt": "2026-05-18T01:00:00Z",
            "payload": payload,
        }
    )


@pytest.mark.asyncio
async def test_room_started_tracks_ai_players() -> None:
    runtime = AgentRuntime(FakeJavaClient(), ScriptedDecisionEngine(), immediate_settings())

    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    assert runtime.room_memory("ABC123").ai_player_ids == {"ai-1"}


@pytest.mark.asyncio
async def test_room_started_and_round_started_track_current_topic() -> None:
    runtime = AgentRuntime(FakeJavaClient(), ScriptedDecisionEngine(), immediate_settings())

    await runtime.handle_event(
        "ABC123",
        event(
            "event-1",
            AgentEventType.ROOM_STARTED,
            {
                "roomCode": "ABC123",
                "topic": {"id": "topic-001", "content": "聊聊最近看的电影"},
                "aiPlayerIds": ["ai-1"],
            },
        ),
    )
    assert runtime.room_memory("ABC123").current_topic == Topic(
        id="topic-001",
        content="聊聊最近看的电影",
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.ROUND_STARTED,
            {
                "roundNumber": 2,
                "topic": {"id": "topic-002", "content": "聊聊最近一次旅行"},
            },
        ),
    )

    assert runtime.room_memory("ABC123").current_topic == Topic(
        id="topic-002",
        content="聊聊最近一次旅行",
    )


@pytest.mark.asyncio
async def test_human_chat_event_can_trigger_ai_message_once() -> None:
    java_client = FakeJavaClient()
    runtime = AgentRuntime(java_client, ScriptedDecisionEngine(), immediate_settings())
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )
    chat_event = event(
        "event-2",
        AgentEventType.CHAT_MESSAGE,
        {
            "messageId": "message-1",
            "senderPlayerId": "human-1",
            "content": "I went hiking last weekend.",
            "createdAt": "2026-05-18T01:00:00Z",
        },
    )

    await runtime.handle_event("ABC123", chat_event)
    await runtime.handle_event("ABC123", chat_event)

    assert java_client.sent_messages == [("ABC123", "ai-1", "That sounds familiar to me.")]


@pytest.mark.asyncio
async def test_chat_decision_context_syncs_topic_from_room_state() -> None:
    java_client = FakeJavaClient()
    java_client.room_state["topic"] = {"id": "topic-003", "content": "聊聊最喜欢的城市"}
    decision_engine = ScriptedDecisionEngine()
    runtime = AgentRuntime(java_client, decision_engine, immediate_settings())
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-1",
                "senderPlayerId": "human-1",
                "content": "I went hiking last weekend.",
                "createdAt": "2026-05-18T01:00:00Z",
            },
        ),
    )

    assert decision_engine.speech_calls[0].current_topic == Topic(
        id="topic-003",
        content="聊聊最喜欢的城市",
    )


@pytest.mark.asyncio
async def test_ai_chat_event_does_not_trigger_reply_to_self() -> None:
    java_client = FakeJavaClient()
    runtime = AgentRuntime(java_client, ScriptedDecisionEngine(), immediate_settings())
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-ai",
                "senderPlayerId": "ai-1",
                "content": "That sounds familiar to me.",
                "createdAt": "2026-05-18T01:00:00Z",
            },
        ),
    )

    assert java_client.sent_messages == []


@pytest.mark.asyncio
async def test_voting_started_casts_vote_without_self_target() -> None:
    java_client = FakeJavaClient()
    java_client.room_state["status"] = "VOTING"
    runtime = AgentRuntime(java_client, ScriptedDecisionEngine(), immediate_settings())
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.VOTING_STARTED,
            {"roomCode": "ABC123", "roundNumber": 1, "candidatePlayerIds": ["human-1", "human-2", "ai-1"]},
        ),
    )

    assert java_client.cast_votes == [("ABC123", "ai-1", "human-1", "Most suspicious.")]


@pytest.mark.asyncio
async def test_vote_submit_failure_is_retried_by_periodic_check() -> None:
    java_client = FakeJavaClient()
    java_client.room_state["status"] = "VOTING"
    java_client.fail_vote_attempts = 1
    runtime = AgentRuntime(java_client, ScriptedDecisionEngine(), immediate_settings())
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.VOTING_STARTED,
            {"roomCode": "ABC123", "roundNumber": 1, "candidatePlayerIds": ["human-1", "human-2", "ai-1"]},
        ),
    )
    await runtime.run_idle_checks()

    assert java_client.cast_votes == [("ABC123", "ai-1", "human-1", "Most suspicious.")]


@pytest.mark.asyncio
async def test_voting_started_state_race_is_retried_by_periodic_check() -> None:
    java_client = FakeJavaClient()
    runtime = AgentRuntime(java_client, ScriptedDecisionEngine(), immediate_settings())
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.VOTING_STARTED,
            {"roomCode": "ABC123", "roundNumber": 1, "candidatePlayerIds": ["human-1", "human-2", "ai-1"]},
        ),
    )
    java_client.room_state["status"] = "VOTING"
    await runtime.run_idle_checks()

    assert java_client.cast_votes == [("ABC123", "ai-1", "human-1", "Most suspicious.")]


@pytest.mark.asyncio
async def test_chat_response_waits_until_typing_delay_expires() -> None:
    java_client = FakeJavaClient()
    runtime = AgentRuntime(
        java_client,
        ScriptedDecisionEngine(),
        Settings(speech_base_delay_seconds=60, speech_typing_seconds_per_char=0, speech_max_delay_seconds=60),
    )
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-1",
                "senderPlayerId": "human-1",
                "content": "I went hiking last weekend.",
                "createdAt": "2026-05-18T01:00:00Z",
            },
        ),
    )

    assert java_client.sent_messages == []
    pending = runtime.room_memory("ABC123").pending_speeches["ai-1"]
    pending.due_at = 0
    await runtime.run_pending_speech_checks()

    assert java_client.sent_messages == [("ABC123", "ai-1", "That sounds familiar to me.")]


@pytest.mark.asyncio
async def test_pending_speech_with_new_human_message_is_reviewed_before_send() -> None:
    java_client = FakeJavaClient()
    decision_engine = ScriptedDecisionEngine()
    decision_engine.pending_speech_decision = PendingSpeechDecision(
        action="send_revised",
        message="I was going to say that too.",
        reason="New human message changed the context.",
    )
    runtime = AgentRuntime(
        java_client,
        decision_engine,
        Settings(speech_base_delay_seconds=60, speech_typing_seconds_per_char=0, speech_max_delay_seconds=60),
    )
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )
    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-1",
                "senderPlayerId": "human-1",
                "content": "I went hiking last weekend.",
                "createdAt": "2026-05-18T01:00:00Z",
            },
        ),
    )
    java_client.messages["messages"].append(
        {
            "messageId": "message-2",
            "senderPlayerId": "human-2",
            "content": "Same here.",
            "createdAt": "2026-05-18T01:00:10Z",
        }
    )

    await runtime.handle_event(
        "ABC123",
        event(
            "event-3",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-2",
                "senderPlayerId": "human-2",
                "content": "Same here.",
                "createdAt": "2026-05-18T01:00:10Z",
            },
        ),
    )
    pending = runtime.room_memory("ABC123").pending_speeches["ai-1"]
    pending.due_at = 0
    await runtime.run_pending_speech_checks()

    assert java_client.sent_messages == [("ABC123", "ai-1", "I was going to say that too.")]
    assert len(decision_engine.pending_speech_reviews) == 1
    assert decision_engine.pending_speech_reviews[0].original_message == "That sounds familiar to me."


@pytest.mark.asyncio
async def test_concurrent_pending_speech_checks_submit_same_draft_once() -> None:
    java_client = SlowSendJavaClient()
    runtime = AgentRuntime(
        java_client,
        ScriptedDecisionEngine(),
        Settings(speech_base_delay_seconds=60, speech_typing_seconds_per_char=0, speech_max_delay_seconds=60),
    )
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )
    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-1",
                "senderPlayerId": "human-1",
                "content": "I went hiking last weekend.",
                "createdAt": "2026-05-18T01:00:00Z",
            },
        ),
    )
    pending = runtime.room_memory("ABC123").pending_speeches["ai-1"]
    pending.due_at = 0

    await asyncio.gather(runtime.run_pending_speech_checks(), runtime.run_pending_speech_checks())

    assert java_client.sent_messages == [("ABC123", "ai-1", "That sounds familiar to me.")]


@pytest.mark.asyncio
async def test_pending_speech_is_discarded_when_room_is_no_longer_chatting() -> None:
    java_client = FakeJavaClient()
    runtime = AgentRuntime(
        java_client,
        ScriptedDecisionEngine(),
        Settings(speech_base_delay_seconds=60, speech_typing_seconds_per_char=0, speech_max_delay_seconds=60),
    )
    await runtime.handle_event(
        "ABC123",
        event("event-1", AgentEventType.ROOM_STARTED, {"roomCode": "ABC123", "aiPlayerIds": ["ai-1"]}),
    )
    await runtime.handle_event(
        "ABC123",
        event(
            "event-2",
            AgentEventType.CHAT_MESSAGE,
            {
                "messageId": "message-1",
                "senderPlayerId": "human-1",
                "content": "I went hiking last weekend.",
                "createdAt": "2026-05-18T01:00:00Z",
            },
        ),
    )
    pending = runtime.room_memory("ABC123").pending_speeches["ai-1"]
    pending.due_at = 0
    java_client.room_state["status"] = "VOTING"

    await runtime.run_pending_speech_checks()

    assert java_client.sent_messages == []
    assert runtime.room_memory("ABC123").pending_speeches == {}
