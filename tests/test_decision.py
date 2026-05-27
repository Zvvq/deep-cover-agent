import pytest

from deep_cover_agent.decision import (
    DecisionContext,
    RuleBasedDecisionEngine,
    SpeechDecision,
    VoteDecision,
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
    assert "只返回 JSON" in speech_prompt
    assert "投票目标" in vote_prompt
    assert "不能选择自己" in vote_prompt


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
