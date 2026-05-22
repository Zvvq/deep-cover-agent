import pytest

from deep_cover_agent.tools import build_agent_query_tools


class FakeJavaClient:
    def __init__(self) -> None:
        self.message_limits = []

    async def get_room_state(self, room_code: str):
        return {"roomCode": room_code, "status": "CHATTING"}

    async def get_messages(self, room_code: str, limit: int = 50):
        self.message_limits.append(limit)
        return {"roomCode": room_code, "messages": [{"messageId": "message-1"}]}

    async def get_votes(self, room_code: str):
        return {"roomCode": room_code, "roundNumber": 1, "candidatePlayerIds": ["player-1"]}


@pytest.mark.asyncio
async def test_query_tools_call_java_internal_apis() -> None:
    java_client = FakeJavaClient()
    tools = {tool.name: tool for tool in build_agent_query_tools(java_client, message_history_limit=20)}

    room_state = await tools["get_room_state"].ainvoke({"room_code": "ABC123"})
    messages = await tools["get_recent_messages"].ainvoke({"room_code": "ABC123", "limit": 10})
    votes = await tools["get_vote_state"].ainvoke({"room_code": "ABC123"})

    assert room_state == {"roomCode": "ABC123", "status": "CHATTING"}
    assert messages == {"roomCode": "ABC123", "messages": [{"messageId": "message-1"}]}
    assert votes == {"roomCode": "ABC123", "roundNumber": 1, "candidatePlayerIds": ["player-1"]}
    assert java_client.message_limits == [10]


@pytest.mark.asyncio
async def test_recent_messages_tool_clamps_limit() -> None:
    java_client = FakeJavaClient()
    tools = {tool.name: tool for tool in build_agent_query_tools(java_client, message_history_limit=20)}

    await tools["get_recent_messages"].ainvoke({"room_code": "ABC123", "limit": 100})
    await tools["get_recent_messages"].ainvoke({"room_code": "ABC123", "limit": 0})

    assert java_client.message_limits == [20, 20]
