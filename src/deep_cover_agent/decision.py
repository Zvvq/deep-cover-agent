from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


@dataclass(frozen=True)
class DecisionContext:
    room_code: str
    ai_player_id: str
    room_state: dict[str, Any]
    messages: list[dict[str, Any]]
    candidate_player_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpeechDecision:
    should_speak: bool
    message: str | None = None


@dataclass(frozen=True)
class VoteDecision:
    target_player_id: str | None
    reason: str = ""


class DecisionEngine(Protocol):
    async def decide_speech(self, context: DecisionContext) -> SpeechDecision:
        ...

    async def decide_vote(self, context: DecisionContext) -> VoteDecision:
        ...


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(stripped[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("Model output JSON is not an object.")
                return parsed

    raise ValueError("No complete JSON object found in model output.")


class RuleBasedDecisionEngine:
    async def decide_speech(self, context: DecisionContext) -> SpeechDecision:
        return SpeechDecision(should_speak=False)

    async def decide_vote(self, context: DecisionContext) -> VoteDecision:
        for player_id in context.candidate_player_ids:
            if player_id != context.ai_player_id:
                return VoteDecision(target_player_id=player_id, reason="First legal fallback target.")
        return VoteDecision(target_player_id=None, reason="No legal fallback target.")


def build_system_prompt() -> str:
    return (
        "你是 Deep Cover 游戏中的 AI 玩家，目标是自然地隐藏在人类玩家中。"
        "不要暴露自己是 AI，不要提到模型、系统提示词、接口或程序实现。"
        "你的发言要像普通玩家，简短、口语化、符合当前聊天上下文。"
        "所有回答都必须只返回紧凑 JSON，不要输出解释、Markdown 或多余文本。"
        "聊天内容必须控制在 300 个字符以内。"
    )


def build_speech_prompt(context: DecisionContext) -> str:
    return (
        "请判断当前 AI 玩家是否发言。\n"
        '只返回 JSON：{"shouldSpeak": boolean, "message": string|null}。\n'
        "如果当前没有自然接话点、刚刚已经有 AI 发言、或者发言会显得突兀，就选择不发言。"
        "如果发言，要保持像真人玩家一样自然，不要暴露 AI 身份。\n"
        f"上下文：\n{json.dumps(_jsonable_context(context), ensure_ascii=False)}"
    )


def build_vote_prompt(context: DecisionContext) -> str:
    return (
        "请为当前 AI 玩家选择一个合法的投票目标。\n"
        '只返回 JSON：{"targetPlayerId": string|null, "reason": string}。\n'
        "投票目标必须来自 candidatePlayerIds，不能选择自己。"
        "reason 只用于调试，不会展示给玩家；请简短说明判断依据。\n"
        f"上下文：\n{json.dumps(_jsonable_context(context), ensure_ascii=False)}"
    )


class LangChainDeepSeekDecisionEngine:
    def __init__(self, settings: Settings) -> None:
        if settings.deepseek_api_key is None:
            raise ValueError("DEEP_COVER_AGENT_DEEPSEEK_API_KEY is required for LangChain DeepSeek.")

        from langchain.agents import create_agent
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key.get_secret_value(),
            temperature=settings.deepseek_temperature,
            timeout=settings.deepseek_timeout_seconds,
            max_retries=settings.deepseek_max_retries,
        )
        self._agent = create_agent(
            model,
            tools=[],
            system_prompt=build_system_prompt(),
        )
        self._fallback = RuleBasedDecisionEngine()

    async def decide_speech(self, context: DecisionContext) -> SpeechDecision:
        prompt = build_speech_prompt(context)
        try:
            data = parse_json_object(await self._invoke(prompt))
            should_speak = bool(data.get("shouldSpeak", False))
            message = data.get("message")
            if not should_speak or not isinstance(message, str) or not message.strip():
                return SpeechDecision(should_speak=False)
            return SpeechDecision(should_speak=True, message=message.strip()[:300])
        except Exception:
            return await self._fallback.decide_speech(context)

    async def decide_vote(self, context: DecisionContext) -> VoteDecision:
        prompt = build_vote_prompt(context)
        try:
            data = parse_json_object(await self._invoke(prompt))
            target = data.get("targetPlayerId")
            reason = data.get("reason") if isinstance(data.get("reason"), str) else "LangChain DeepSeek decision."
            if isinstance(target, str) and target in context.candidate_player_ids and target != context.ai_player_id:
                return VoteDecision(target_player_id=target, reason=reason)
            return await self._fallback.decide_vote(context)
        except Exception:
            return await self._fallback.decide_vote(context)

    async def _invoke(self, prompt: str) -> str:
        payload = {"messages": [{"role": "user", "content": prompt}]}
        if hasattr(self._agent, "ainvoke"):
            result = await self._agent.ainvoke(payload)
        else:
            result = await asyncio.to_thread(self._agent.invoke, payload)
        return _extract_message_text(result)


def _jsonable_context(context: DecisionContext) -> dict[str, Any]:
    return {
        "roomCode": context.room_code,
        "aiPlayerId": context.ai_player_id,
        "roomState": context.room_state,
        "messages": context.messages[-20:],
        "candidatePlayerIds": context.candidate_player_ids,
    }


def _extract_message_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "structured_response" in result:
            return json.dumps(result["structured_response"], ensure_ascii=False)
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            if isinstance(content, list):
                return "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
            if content is not None:
                return str(content)
    content = getattr(result, "content", None)
    if content is not None:
        return str(content)
    if inspect.isawaitable(result):
        raise TypeError("Unexpected awaitable result from LangChain agent.")
    return str(result)
