import logging
import sys
from importlib.resources import files
from types import SimpleNamespace

import pytest
import yaml

from deep_cover_agent.config import Settings
from deep_cover_agent.decision import (
    DecisionContext,
    LangChainDeepSeekDecisionEngine,
    PendingSpeechContext,
    RuleBasedDecisionEngine,
    SpeechDecision,
    VoteDecision,
    build_pending_speech_review_prompt,
    build_persona_prompt,
    build_speech_prompt,
    build_system_prompt,
    build_vote_prompt,
    parse_json_object,
)
from deep_cover_agent.models import Topic


def test_parse_json_object_accepts_fenced_model_output() -> None:
    parsed = parse_json_object(
        """
        I will answer in JSON.
        ```json
        {"shouldSpeak": true, "message": "I remember something like that."}
        ```
        """
    )

    assert parsed == {"shouldSpeak": True, "message": "I remember something like that."}


def test_langchain_prompts_are_written_in_chinese() -> None:
    context = DecisionContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=[],
        current_topic=Topic(id="topic-001", content="聊聊周末计划"),
        candidate_player_ids=["human-1"],
    )

    system_prompt = build_system_prompt()
    speech_prompt = build_speech_prompt(context)
    vote_prompt = build_vote_prompt(context)

    assert "你是 Deep Cover 游戏中的 AI 玩家" in system_prompt
    assert "只读查询工具" in system_prompt
    assert "是否应该发言" in speech_prompt
    assert "当前话题：聊聊周末计划" in speech_prompt
    assert "请围绕当前话题自然发言" in speech_prompt
    assert "不要为了贴合当前话题硬拉回话题" in speech_prompt
    assert '"messages": string[]' in speech_prompt
    assert "最多 3 段" in speech_prompt
    assert "只返回 JSON" in speech_prompt
    assert "投票目标" in vote_prompt
    assert "不能选择自己" in vote_prompt


def test_speech_prompts_use_distinct_casual_blunt_persona() -> None:
    context = DecisionContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=[{"senderPlayerId": "human-1", "content": "我今天碰到个特别离谱的事"}],
        current_topic=Topic(id="topic-001", content="聊聊最近遇到的离谱事"),
    )
    pending_context = PendingSpeechContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=context.messages,
        original_message="卧槽这也太离谱了",
        remaining_messages=["这也能发生啊"],
        current_topic=context.current_topic,
    )

    combined_prompt = "\n".join(
        [
            build_system_prompt(),
            build_speech_prompt(context),
            build_pending_speech_review_prompt(pending_context),
        ]
    )

    assert "casual_blunt" in combined_prompt
    assert "嘴比较直" in combined_prompt
    assert "不是客服式、标准答案式语气" in combined_prompt
    assert "不知道怎么措辞" in combined_prompt
    assert "卧槽" in combined_prompt
    assert "牛逼" in combined_prompt
    assert "我草" in combined_prompt
    assert "不要连续多次使用同一种口头禅" in combined_prompt
    assert "不要攻击玩家本人" in combined_prompt
    assert "不要使用歧视性、仇恨或性骚扰类词汇" in combined_prompt


def test_speech_prompt_uses_room_bound_persona() -> None:
    context = DecisionContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=[],
        persona_name="lowkey_blunt",
        persona_prompt="房间固定人格：低调直接，少解释，多用短句。",
    )

    prompt = build_speech_prompt(context)

    assert "当前房间固定人格：lowkey_blunt" in prompt
    assert "房间固定人格：低调直接，少解释，多用短句。" in prompt


def test_persona_prompt_is_loaded_from_yaml_config_file() -> None:
    prompt_path = files("deep_cover_agent").joinpath("prompts/persona_prompt.yml")

    assert prompt_path.is_file()
    config = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    personas = config.get("personas", [])

    assert len(personas) == 3
    assert [persona["name"] for persona in personas] == ["casual_blunt", "lowkey_blunt", "quick_reactor"]
    assert build_persona_prompt() == personas[0]["prompt"].strip()
    assert all("不要攻击玩家本人" in persona["prompt"] for persona in personas)


def test_pending_speech_review_prompt_prioritizes_latest_messages() -> None:
    context = PendingSpeechContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=[],
        original_message="如果只能吃一种主食，那我选米饭。",
        sent_messages=["如果只能吃一种主食"],
        remaining_messages=["那我选米饭。"],
        new_messages=[{"senderPlayerId": "human-1", "content": "你们投的谁？"}],
        current_topic=Topic(id="topic-001", content="如果今晚只能吃一种主食，你会选什么？"),
    )

    prompt = build_pending_speech_review_prompt(context)

    assert "优先根据最新消息判断" in prompt
    assert "不要为了贴合当前话题硬拉回话题" in prompt
    assert "remainingMessages" in prompt


def test_langchain_deepseek_disables_thinking_mode_by_default(monkeypatch) -> None:
    captured = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    def fake_create_agent(model, tools, system_prompt):
        captured["agent_model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        return object()

    monkeypatch.setitem(sys.modules, "langchain.agents", SimpleNamespace(create_agent=fake_create_agent))
    monkeypatch.setitem(sys.modules, "langchain_deepseek", SimpleNamespace(ChatDeepSeek=FakeChatDeepSeek))

    LangChainDeepSeekDecisionEngine(Settings(_env_file=None, deepseek_api_key="test-key"))

    assert captured["model_kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_langchain_deepseek_passes_custom_base_url(monkeypatch) -> None:
    captured = {}

    def fake_create_agent(model, tools, system_prompt):
        captured["agent_model"] = model
        return object()

    monkeypatch.setitem(sys.modules, "langchain.agents", SimpleNamespace(create_agent=fake_create_agent))

    LangChainDeepSeekDecisionEngine(
        Settings(
            _env_file=None,
            deepseek_api_key="test-key",
            deepseek_base_url="https://relay.example/v1",
        )
    )

    assert str(captured["agent_model"].root_client.base_url) == "https://relay.example/v1/"
    assert captured["agent_model"].model_kwargs == {}


@pytest.mark.asyncio
async def test_langchain_pending_speech_review_failure_discards_stale_draft() -> None:
    class BrokenReviewEngine(LangChainDeepSeekDecisionEngine):
        def __init__(self) -> None:
            self._fallback = RuleBasedDecisionEngine()

        async def _invoke(self, prompt: str) -> str:
            raise RuntimeError("model review failed")

    engine = BrokenReviewEngine()
    context = PendingSpeechContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=[],
        original_message="如果只能吃一种主食，那我选米饭。",
        sent_messages=["如果只能吃一种主食"],
        remaining_messages=["那我选米饭。"],
        new_messages=[{"senderPlayerId": "human-1", "content": "你们投的谁？"}],
        current_topic=Topic(id="topic-001", content="如果今晚只能吃一种主食，你会选什么？"),
    )

    decision = await engine.review_pending_speech(context)

    assert decision.action == "discard"
    assert "复核失败" in decision.reason


@pytest.mark.asyncio
async def test_langchain_speech_decision_accepts_segmented_messages() -> None:
    class SegmentedSpeechEngine(LangChainDeepSeekDecisionEngine):
        def __init__(self) -> None:
            self._fallback = RuleBasedDecisionEngine()

        async def _invoke(self, prompt: str) -> str:
            return '{"shouldSpeak": true, "messages": ["观察力吧", "感觉能少踩坑"]}'

    engine = SegmentedSpeechEngine()

    decision = await engine.decide_speech(
        DecisionContext(
            room_code="ABC123",
            ai_player_id="ai-1",
            room_state={"status": "CHATTING"},
            messages=[],
        )
    )

    assert decision.should_speak is True
    assert decision.messages == ("观察力吧", "感觉能少踩坑")
    assert decision.message == "观察力吧"


@pytest.mark.asyncio
async def test_langchain_failures_log_summary_without_traceback(caplog) -> None:
    class BrokenSpeechEngine(LangChainDeepSeekDecisionEngine):
        def __init__(self) -> None:
            self._fallback = RuleBasedDecisionEngine()

        async def _invoke(self, prompt: str) -> str:
            raise RuntimeError("model failed")

    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    engine = BrokenSpeechEngine()

    await engine.decide_speech(
        DecisionContext(
            room_code="ABC123",
            ai_player_id="ai-1",
            room_state={"status": "CHATTING"},
            messages=[],
        )
    )

    model_records = [record for record in caplog.records if "[模型]" in record.getMessage()]
    assert model_records
    assert all(record.exc_info is None for record in model_records)


@pytest.mark.asyncio
async def test_rule_based_vote_chooses_first_non_self_candidate() -> None:
    engine = RuleBasedDecisionEngine()
    context = DecisionContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "VOTING"},
        messages=[],
        candidate_player_ids=["ai-1", "human-2", "human-1"],
    )

    decision = await engine.decide_vote(context)

    assert isinstance(decision, VoteDecision)
    assert decision.target_player_id == "human-2"


@pytest.mark.asyncio
async def test_rule_based_speech_stays_quiet_by_default() -> None:
    engine = RuleBasedDecisionEngine()
    context = DecisionContext(
        room_code="ABC123",
        ai_player_id="ai-1",
        room_state={"status": "CHATTING"},
        messages=[],
        candidate_player_ids=[],
    )

    decision = await engine.decide_speech(context)

    assert isinstance(decision, SpeechDecision)
    assert decision.should_speak is False
