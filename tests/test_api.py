import logging

from fastapi.testclient import TestClient

from deep_cover_agent.api import build_runtime, create_app
from deep_cover_agent.config import Settings


class RecordingRuntime:
    def __init__(self) -> None:
        self.events = []

    async def handle_event(self, room_code, event):
        self.events.append((room_code, event))

    async def shutdown(self):
        return None


def agent_event(room_code: str = "ABC123") -> dict:
    return {
        "eventId": "event-1",
        "type": "ROOM_STARTED",
        "roomCode": room_code,
        "createdAt": "2026-05-18T01:00:00Z",
        "payload": {"roomCode": room_code, "aiPlayerIds": ["ai-1"]},
    }


def test_build_runtime_passes_java_client_to_langchain_engine(monkeypatch) -> None:
    captured = {}

    class FakeLangChainEngine:
        def __init__(self, settings, java_client):
            captured["settings"] = settings
            captured["java_client"] = java_client

    monkeypatch.setattr("deep_cover_agent.api.LangChainDeepSeekDecisionEngine", FakeLangChainEngine)

    runtime = build_runtime(Settings(enable_langchain=True, deepseek_api_key="test-key"))

    assert captured["settings"].deepseek_api_key is not None
    assert captured["java_client"] is runtime._java_client


def test_rejects_event_without_internal_secret() -> None:
    runtime = RecordingRuntime()
    app = create_app(Settings(internal_agent_secret="test-secret"), runtime)
    client = TestClient(app)

    response = client.post("/agent/rooms/ABC123/events", json=agent_event())

    assert response.status_code == 401
    assert runtime.events == []


def test_accepts_event_with_internal_secret() -> None:
    runtime = RecordingRuntime()
    app = create_app(Settings(internal_agent_secret="test-secret"), runtime)
    client = TestClient(app)

    response = client.post(
        "/agent/rooms/ABC123/events",
        json=agent_event(),
        headers={"X-Internal-Agent-Secret": "test-secret"},
    )

    assert response.status_code == 202
    assert response.json() == {"accepted": True}
    assert len(runtime.events) == 1
    assert runtime.events[0][0] == "ABC123"
    assert runtime.events[0][1].event_id == "event-1"


def test_accepts_event_logs_chinese_business_messages(caplog) -> None:
    runtime = RecordingRuntime()
    app = create_app(Settings(internal_agent_secret="test-secret"), runtime)
    client = TestClient(app)
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    response = client.post(
        "/agent/rooms/ABC123/events",
        json=agent_event(),
        headers={"X-Internal-Agent-Secret": "test-secret"},
    )

    assert response.status_code == 202
    messages = [record.getMessage() for record in caplog.records]
    assert any("收到 Java 事件" in message and "房间=ABC123" in message for message in messages)
    assert any("事件处理完成" in message and "事件ID=event-1" in message for message in messages)


def test_accepts_event_with_jackson_array_timestamp() -> None:
    runtime = RecordingRuntime()
    app = create_app(Settings(internal_agent_secret="test-secret"), runtime)
    client = TestClient(app)
    event = agent_event()
    event["createdAt"] = [2026, 5, 18, 1, 0, 0, 123456789]

    response = client.post(
        "/agent/rooms/ABC123/events",
        json=event,
        headers={"X-Internal-Agent-Secret": "test-secret"},
    )

    assert response.status_code == 202
    assert len(runtime.events) == 1
    assert runtime.events[0][1].created_at.microsecond == 123456


def test_rejects_event_when_path_room_does_not_match_body() -> None:
    runtime = RecordingRuntime()
    app = create_app(Settings(internal_agent_secret="test-secret"), runtime)
    client = TestClient(app)

    response = client.post(
        "/agent/rooms/ABC123/events",
        json=agent_event("OTHER1"),
        headers={"X-Internal-Agent-Secret": "test-secret"},
    )

    assert response.status_code == 400
    assert runtime.events == []
