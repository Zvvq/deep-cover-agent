import json

import httpx
import pytest

from deep_cover_agent.java_client import JavaAgentClient


@pytest.mark.asyncio
async def test_java_client_sends_secret_header_and_fetches_state() -> None:
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={
                "roomCode": "ABC123",
                "status": "CHATTING",
                "roundNumber": 1,
                "aliveHumanCount": 2,
                "aliveAiCount": 1,
                "players": [],
            },
        )

    client = JavaAgentClient(
        base_url="http://java.local",
        internal_secret="test-secret",
        transport=httpx.MockTransport(handler),
    )

    response = await client.get_room_state("ABC123")

    assert response["roomCode"] == "ABC123"
    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/api/internal/agent/rooms/ABC123/state"
    assert seen_requests[0].headers["X-Internal-Agent-Secret"] == "test-secret"
    await client.aclose()


@pytest.mark.asyncio
async def test_java_client_posts_vote_with_reason() -> None:
    seen_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"roomCode": "ABC123", "roundNumber": 1, "settled": False})

    client = JavaAgentClient(
        base_url="http://java.local",
        internal_secret="test-secret",
        transport=httpx.MockTransport(handler),
    )

    await client.cast_vote("ABC123", "ai-1", "human-1", "Most suspicious.")

    assert seen_body == {
        "aiPlayerId": "ai-1",
        "targetPlayerId": "human-1",
        "reason": "Most suspicious.",
    }
    await client.aclose()
