from deep_cover_agent.models import AgentRoomStateResponse, Topic


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
