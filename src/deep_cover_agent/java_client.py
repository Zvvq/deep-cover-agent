from typing import Any

import httpx


class JavaAgentClient:
    def __init__(
        self,
        base_url: str,
        internal_secret: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"X-Internal-Agent-Secret": internal_secret},
        )

    async def get_room_state(self, room_code: str) -> dict[str, Any]:
        response = await self._client.get(f"/api/internal/agent/rooms/{room_code}/state")
        response.raise_for_status()
        return response.json()

    async def get_messages(self, room_code: str, limit: int = 50) -> dict[str, Any]:
        response = await self._client.get(
            f"/api/internal/agent/rooms/{room_code}/messages",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()

    async def get_votes(self, room_code: str) -> dict[str, Any]:
        response = await self._client.get(f"/api/internal/agent/rooms/{room_code}/votes")
        response.raise_for_status()
        return response.json()

    async def send_message(self, room_code: str, ai_player_id: str, content: str) -> dict[str, Any]:
        response = await self._client.post(
            f"/api/internal/agent/rooms/{room_code}/messages",
            json={"aiPlayerId": ai_player_id, "content": content},
        )
        response.raise_for_status()
        return response.json()

    async def cast_vote(
        self,
        room_code: str,
        ai_player_id: str,
        target_player_id: str,
        reason: str,
    ) -> dict[str, Any]:
        response = await self._client.post(
            f"/api/internal/agent/rooms/{room_code}/votes",
            json={
                "aiPlayerId": ai_player_id,
                "targetPlayerId": target_player_id,
                "reason": reason,
            },
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
