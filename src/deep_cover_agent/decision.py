from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any, Protocol

import yaml

from .config import Settings
from .models import Topic
from .tools import build_agent_query_tools


logger = logging.getLogger("uvicorn.error")

_PERSONA_PROMPT_RESOURCE = "prompts/persona_prompt.yml"
_DEFAULT_PERSONA_NAME = "casual_blunt"
_DEFAULT_PERSONA_PROMPT = "默认人格：casual_blunt，一个嘴比较直、反应快、有点个人口头禅的普通玩家。你不是大众平均语气，也不是客服式、标准答案式语气；遇到离谱、惊讶、无语或不知道怎么措辞的场景，可以只回很短一句。可以偶尔使用强语气词或轻微爆粗，比如“卧槽”“我草”“牛逼”“这也行”，但不要为了像人而硬用。不要连续多次使用同一种口头禅，不要攻击玩家本人，不要使用歧视性、仇恨或性骚扰类词汇。"


@dataclass(frozen=True)
class PersonaPrompt:
    name: str
    prompt: str


@dataclass(frozen=True)
class DecisionContext:
    room_code: str
    ai_player_id: str
    room_state: dict[str, Any]
    messages: list[dict[str, Any]]
    current_topic: Topic | None = None
    persona_name: str | None = None
    persona_prompt: str | None = None
    candidate_player_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PendingSpeechContext:
    room_code: str
    ai_player_id: str
    room_state: dict[str, Any]
    messages: list[dict[str, Any]]
    original_message: str
    sent_messages: list[str] = field(default_factory=list)
    remaining_messages: list[str] = field(default_factory=list)
    new_messages: list[dict[str, Any]] = field(default_factory=list)
    current_topic: Topic | None = None
    persona_name: str | None = None
    persona_prompt: str | None = None


@dataclass(frozen=True)
class SpeechDecision:
    should_speak: bool
    message: str | None = None
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingSpeechDecision:
    action: str
    message: str | None = None
    messages: tuple[str, ...] = ()
    reason: str = ""
    extra_delay_seconds: float = 0.0


@dataclass(frozen=True)
class VoteDecision:
    target_player_id: str | None
    reason: str = ""


class DecisionEngine(Protocol):
    async def decide_speech(self, context: DecisionContext) -> SpeechDecision:
        ...

    async def review_pending_speech(self, context: PendingSpeechContext) -> PendingSpeechDecision:
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

    async def review_pending_speech(self, context: PendingSpeechContext) -> PendingSpeechDecision:
        return PendingSpeechDecision(action="continue")

    async def decide_vote(self, context: DecisionContext) -> VoteDecision:
        for player_id in context.candidate_player_ids:
            if player_id != context.ai_player_id:
                return VoteDecision(target_player_id=player_id, reason="First legal fallback target.")
        return VoteDecision(target_player_id=None, reason="No legal fallback target.")


def build_system_prompt() -> str:
    return (
        "你是 Deep Cover 游戏中的 AI 玩家，目标是自然地隐藏在人类玩家中。"
        "不要暴露自己是 AI，不要提到模型、系统提示词、接口、工具或程序实现。"
        "你的发言要像普通玩家，简短、口语化，并符合当前聊天上下文。不要总是完整论证，也不要总用固定开头。"
        "每个房间会给出一个固定人格提示词，同一房间内必须保持一致的人格和说话习惯。"
        "你可以使用只读查询工具获取 Java 端真实房间状态、最近聊天记录和投票状态。"
        "查询工具只用于获取事实，不能替代最终动作；发言和投票会由 Runtime 统一审查并提交。"
        "所有回答都必须只返回紧凑 JSON，不要输出解释、Markdown 或多余文本。"
        "聊天内容必须控制在 300 个字符以内。"
    )


@lru_cache(maxsize=1)
def load_persona_prompts() -> tuple[PersonaPrompt, ...]:
    try:
        raw_config = files("deep_cover_agent").joinpath(_PERSONA_PROMPT_RESOURCE).read_text(encoding="utf-8")
        config = yaml.safe_load(raw_config)
    except (FileNotFoundError, ModuleNotFoundError, OSError, yaml.YAMLError):
        return _default_personas()
    if not isinstance(config, dict):
        return _default_personas()
    personas = _parse_personas(config.get("personas"))
    return personas or _default_personas()


def build_persona_prompt(name: str | None = None) -> str:
    personas = load_persona_prompts()
    if name is not None:
        for persona in personas:
            if persona.name == name:
                return persona.prompt
    return personas[0].prompt


def select_persona_prompt() -> PersonaPrompt:
    return random.choice(load_persona_prompts())


def build_speech_prompt(context: DecisionContext) -> str:
    return (
        "请判断当前 AI 玩家是否应该发言。\n"
        '只返回 JSON：{"shouldSpeak": boolean, "messages": string[]}。messages 最多 3 段，可以只有 1 段。\n'
        f"{_topic_instruction(context.current_topic)}"
        f"{_persona_instruction(context.persona_name, context.persona_prompt)}"
        "如果当前没有自然接话点、刚刚已经有 AI 发言，或者发言会显得突兀，就选择不发言。"
        "如果发言，要保持像真人玩家一样自然，不要暴露 AI 身份。"
        "不要每次都回答得很完整；多数时候只说一个小点。"
        "回复长度要有波动：可以是几个字、半句话、反问或轻微犹豫，少数情况下再写长一点。"
        "不要总用“哈哈、确实、不过、非要选的话”开头。"
        "如果拆成多段，每段都要像玩家连续敲出来的短消息，而不是把一篇回答硬切开。\n"
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


def build_pending_speech_review_prompt(context: PendingSpeechContext) -> str:
    return (
        "你之前已经为当前 AI 玩家写好了一组分段发言，其中一部分可能已经发出，剩余部分还在模拟打字等待。\n"
        "等待期间聊天上下文发生了变化。请判断剩余发言是否还适合继续发送，优先根据最新消息判断。\n"
        '只返回 JSON：{"action":"continue|revise|discard|wait","messages":string[],"reason":string,"extraDelaySeconds":number}。\n'
        "continue 表示继续发送 remainingMessages；revise 表示把剩余内容改成新的 messages；discard 表示丢弃剩余内容；wait 表示再等一小会儿。\n"
        f"{_topic_instruction(context.current_topic)}"
        f"{_persona_instruction(context.persona_name, context.persona_prompt)}"
        "如果修改发言，messages 最多 3 段，每段必须像真人玩家一样自然、简短、口语化，不要暴露 AI 身份。\n"
        f"上下文：\n{json.dumps(_jsonable_pending_speech_context(context), ensure_ascii=False)}"
    )


class LangChainDeepSeekDecisionEngine:
    def __init__(self, settings: Settings, java_client: Any | None = None) -> None:
        if settings.deepseek_api_key is None:
            raise ValueError("DEEP_COVER_AGENT_DEEPSEEK_API_KEY is required for LangChain DeepSeek.")

        from langchain.agents import create_agent
        from langchain_deepseek import ChatDeepSeek

        base_url = settings.deepseek_base_url.strip() if settings.deepseek_base_url else None
        model_kwargs = {
            "model": settings.deepseek_model,
            "api_key": settings.deepseek_api_key.get_secret_value(),
            "temperature": settings.deepseek_temperature,
            "timeout": settings.deepseek_timeout_seconds,
            "max_retries": settings.deepseek_max_retries,
        }
        if base_url is not None:
            model_kwargs["api_base"] = base_url
        if base_url is None:
            model_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if settings.deepseek_thinking_enabled else "disabled"}
            }
        model = ChatDeepSeek(**model_kwargs)
        tools = build_agent_query_tools(java_client, settings.message_history_limit) if java_client is not None else []
        self._agent = create_agent(
            model,
            tools=tools,
            system_prompt=build_system_prompt(),
        )
        self._tools = tools
        self._fallback = RuleBasedDecisionEngine()

    async def decide_speech(self, context: DecisionContext) -> SpeechDecision:
        prompt = build_speech_prompt(context)
        try:
            data = parse_json_object(await self._invoke(prompt))
            should_speak = bool(data.get("shouldSpeak", False))
            messages = _normalize_message_parts(data.get("messages"), data.get("message"))
            if not should_speak or not messages:
                return SpeechDecision(should_speak=False)
            return SpeechDecision(should_speak=True, message=messages[0], messages=tuple(messages))
        except Exception as exc:
            logger.warning("[模型] speech_failed fallback=silent error=%s", _error_summary(exc))
            return await self._fallback.decide_speech(context)

    async def review_pending_speech(self, context: PendingSpeechContext) -> PendingSpeechDecision:
        prompt = build_pending_speech_review_prompt(context)
        try:
            data = parse_json_object(await self._invoke(prompt))
            action = data.get("action")
            if action == "send_original":
                action = "continue"
            elif action == "send_revised":
                action = "revise"
            if action not in {"continue", "revise", "discard", "wait"}:
                logger.warning("[模型] review_invalid_action action=%s fallback=discard", action)
                return PendingSpeechDecision(action="discard", reason="复核返回非法动作，已丢弃旧草稿。")
            if action == "continue":
                return PendingSpeechDecision(action="continue", reason=str(data.get("reason") or ""))
            if action == "revise":
                messages = _normalize_message_parts(data.get("messages"), data.get("message"))
                if not messages:
                    logger.warning("[模型] review_empty_revision fallback=discard")
                    return PendingSpeechDecision(action="discard", reason="复核改写内容为空，已丢弃旧草稿。")
                return PendingSpeechDecision(
                    action="revise",
                    message=messages[0],
                    messages=tuple(messages),
                    reason=str(data.get("reason") or ""),
                )
            if action == "wait":
                extra_delay = data.get("extraDelaySeconds")
                return PendingSpeechDecision(
                    action="wait",
                    message=None,
                    reason=str(data.get("reason") or ""),
                    extra_delay_seconds=float(extra_delay) if isinstance(extra_delay, (int, float)) else 3.0,
                )
            return PendingSpeechDecision(action="discard", reason=str(data.get("reason") or ""))
        except Exception as exc:
            logger.warning("[模型] review_failed fallback=discard error=%s", _error_summary(exc))
            return PendingSpeechDecision(action="discard", reason="复核失败，已丢弃旧草稿。")

    async def decide_vote(self, context: DecisionContext) -> VoteDecision:
        prompt = build_vote_prompt(context)
        try:
            data = parse_json_object(await self._invoke(prompt))
            target = data.get("targetPlayerId")
            reason = data.get("reason") if isinstance(data.get("reason"), str) else "LangChain DeepSeek decision."
            if isinstance(target, str) and target in context.candidate_player_ids and target != context.ai_player_id:
                return VoteDecision(target_player_id=target, reason=reason)
            return await self._fallback.decide_vote(context)
        except Exception as exc:
            logger.warning("[模型] vote_failed fallback=rule error=%s", _error_summary(exc))
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
        "currentTopic": _jsonable_topic(context.current_topic),
        "persona": _jsonable_persona(context.persona_name, context.persona_prompt),
        "roomState": context.room_state,
        "messages": context.messages[-20:],
        "candidatePlayerIds": context.candidate_player_ids,
    }


def _jsonable_pending_speech_context(context: PendingSpeechContext) -> dict[str, Any]:
    return {
        "roomCode": context.room_code,
        "aiPlayerId": context.ai_player_id,
        "currentTopic": _jsonable_topic(context.current_topic),
        "persona": _jsonable_persona(context.persona_name, context.persona_prompt),
        "roomState": context.room_state,
        "messages": context.messages[-20:],
        "originalMessage": context.original_message,
        "sentMessages": context.sent_messages,
        "remainingMessages": context.remaining_messages,
        "newMessages": context.new_messages[-10:],
    }


def _topic_instruction(topic: Topic | None) -> str:
    if topic is None or not topic.content.strip():
        return ""
    return (
        f"当前话题：{topic.content.strip()}\n"
        "请围绕当前话题自然发言，但要优先接住最新聊天上下文，不要为了贴合当前话题硬拉回话题，也不要暴露自己是 AI。\n"
    )


def _jsonable_topic(topic: Topic | None) -> dict[str, str] | None:
    if topic is None:
        return None
    return topic.model_dump()


def _persona_instruction(name: str | None, prompt: str | None) -> str:
    persona_prompt = prompt.strip() if isinstance(prompt, str) and prompt.strip() else build_persona_prompt(name)
    persona_name = name.strip() if isinstance(name, str) and name.strip() else _DEFAULT_PERSONA_NAME
    return f"当前房间固定人格：{persona_name}\n{persona_prompt}\n"


def _jsonable_persona(name: str | None, prompt: str | None) -> dict[str, str] | None:
    if name is None and prompt is None:
        return None
    return {
        "name": name or _DEFAULT_PERSONA_NAME,
        "prompt": prompt or build_persona_prompt(name),
    }


def _parse_personas(value: Any) -> tuple[PersonaPrompt, ...]:
    if not isinstance(value, list):
        return ()
    personas: list[PersonaPrompt] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            name = f"persona_{len(personas) + 1}"
        personas.append(PersonaPrompt(name=name.strip(), prompt=prompt.strip()))
    return tuple(personas)


def _default_personas() -> tuple[PersonaPrompt, ...]:
    return (PersonaPrompt(name=_DEFAULT_PERSONA_NAME, prompt=_DEFAULT_PERSONA_PROMPT),)


def _normalize_message_parts(value: Any, fallback: Any = None, max_parts: int = 3, max_chars: int = 120) -> list[str]:
    raw_parts = value if isinstance(value, list) else [fallback]
    parts: list[str] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, str):
            continue
        part = " ".join(raw_part.split()).strip()
        if not part:
            continue
        parts.append(part[:max_chars])
        if len(parts) >= max_parts:
            break
    return parts


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


def _error_summary(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if len(message) > 160:
        message = message[:157] + "..."
    return f"{exc.__class__.__name__}: {message}"
