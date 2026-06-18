from deep_cover_agent.models import AgentEvent, AgentRoomStateResponse, Topic


def test_room_state_response_accepts_current_topic() -> None:
    room_state = AgentRoomStateResponse.model_validate(
        {
            "roomCode": "ABC123",
            "status": "CHATTING",
            "roundNumber": 2,
            "aliveHumanCount": 2,
            "aliveAiCount": 1,
            "topic": {
                "id": "topic-002",
                "content": "聊聊最近一次旅行",
            },
            "players": [],
        }
    )

    assert isinstance(room_state.topic, Topic)
    assert room_state.topic.id == "topic-002"
    assert room_state.topic.content == "聊聊最近一次旅行"


def test_room_state_response_accepts_word_undercover_mode_and_describing_status() -> None:
    room_state = AgentRoomStateResponse.model_validate(
        {
            "roomCode": "ABC123",
            "gameMode": "WORD_UNDERCOVER",
            "status": "DESCRIBING",
            "roundNumber": 1,
            "aliveHumanCount": 2,
            "aliveAiCount": 1,
            "players": [],
        }
    )

    assert room_state.game_mode == "WORD_UNDERCOVER"
    assert room_state.status == "DESCRIBING"


def test_agent_event_accepts_word_undercover_event_types() -> None:
    for event_type in ["WORD_ROUND_STARTED", "WORD_DESCRIPTION_SUBMITTED"]:
        event = AgentEvent.model_validate(
            {
                "eventId": f"event-{event_type}",
                "type": event_type,
                "roomCode": "ABC123",
                "createdAt": "2026-06-10T01:22:18.910027700Z",
                "payload": {"roundNumber": 1},
            }
        )

        assert event.type == event_type
