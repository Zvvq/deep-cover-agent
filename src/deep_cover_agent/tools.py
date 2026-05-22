from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool


def build_agent_query_tools(java_client: Any, message_history_limit: int = 50) -> list[BaseTool]:
    """Build read-only LangChain tools backed by Java internal agent APIs."""

    @tool
    async def get_room_state(room_code: str) -> dict[str, Any]:
        """查询 Java 端当前真实房间状态，包括阶段、轮次、玩家列表和存活情况。"""
        return await java_client.get_room_state(room_code)

    @tool
    async def get_recent_messages(room_code: str, limit: int | None = None) -> dict[str, Any]:
        """查询 Java 端最近聊天记录。limit 会被限制在 Agent 配置的消息历史上限内。"""
        normalized_limit = _normalize_limit(limit, message_history_limit)
        return await java_client.get_messages(room_code, normalized_limit)

    @tool
    async def get_vote_state(room_code: str) -> dict[str, Any]:
        """查询 Java 端当前投票状态，包括轮次、已提交票数、所需票数和候选玩家。"""
        return await java_client.get_votes(room_code)

    return [get_room_state, get_recent_messages, get_vote_state]


def _normalize_limit(limit: int | None, message_history_limit: int) -> int:
    if limit is None or limit <= 0:
        return message_history_limit
    return min(limit, message_history_limit)
