# Deep Cover Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python event-driven Agent Runtime that receives Java game events, reasons with LangChain + DeepSeek, and submits AI chat/vote actions back to Java internal endpoints.

**Architecture:** FastAPI exposes only the Java event push endpoint. An in-memory room runtime deduplicates events, tracks AI player IDs, queries Java for authoritative state, and delegates speech/vote choices to a decision engine. Java remains the rule authority; Python only proposes actions.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx, LangChain v1.x, langchain-deepseek, pytest.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_api.py`
- Create: `tests/test_runtime.py`
- Create: `tests/test_java_client.py`
- Create: `tests/test_decision.py`

- [ ] Write failing tests for `POST /agent/rooms/{roomCode}/events` authentication and room-code validation.
- [ ] Write failing tests for runtime event deduplication, human chat response, AI self-message suppression, and voting without self-targeting.
- [ ] Write failing tests for Java internal client paths and `X-Internal-Agent-Secret` headers.
- [ ] Write failing tests for JSON decision parsing and deterministic fallback voting.
- [ ] Run `pytest -q`; expected result before implementation is import failures for `deep_cover_agent`.

### Task 2: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/deep_cover_agent/__init__.py`
- Create: `src/deep_cover_agent/models.py`
- Create: `src/deep_cover_agent/config.py`

- [ ] Add package metadata and dependencies.
- [ ] Define settings for Java base URL, internal secret, DeepSeek API key/model, and idle intervals.
- [ ] Define Pydantic models for Java event envelopes, payloads, state responses, message responses, vote responses, and action commands.
- [ ] Run focused model/import tests until imports pass.

### Task 3: Java Client

**Files:**
- Create: `src/deep_cover_agent/java_client.py`

- [ ] Implement `JavaAgentClient` with async `httpx`.
- [ ] Add methods for room state, recent messages, vote state, AI message submission, and AI vote submission.
- [ ] Ensure every call sends `X-Internal-Agent-Secret`.
- [ ] Run `pytest tests/test_java_client.py -q`.

### Task 4: Decision Engine

**Files:**
- Create: `src/deep_cover_agent/decision.py`

- [ ] Define decision context and speech/vote decision value objects.
- [ ] Implement `RuleBasedDecisionEngine` for deterministic fallback behavior.
- [ ] Implement `LangChainDeepSeekDecisionEngine` using `langchain.agents.create_agent` and `langchain_deepseek.ChatDeepSeek`.
- [ ] Parse model output as strict JSON, with fallback decisions on invalid output.
- [ ] Run `pytest tests/test_decision.py -q`.

### Task 5: Runtime And API

**Files:**
- Create: `src/deep_cover_agent/runtime.py`
- Create: `src/deep_cover_agent/api.py`
- Create: `src/deep_cover_agent/main.py`
- Create: `src/deep_cover_agent/__main__.py`

- [ ] Implement per-room memory for AI player IDs, processed event IDs, attempted votes, and idle timing.
- [ ] Implement event handlers for room start, chat messages, voting start, round start, game end, and room destroyed.
- [ ] On chat events, query Java state/messages before acting and only respond for alive AI players during `CHATTING`.
- [ ] On voting events, query Java state/messages/votes before acting and exclude the AI player itself from targets.
- [ ] Add FastAPI app factory with internal-secret authentication.
- [ ] Add startup/shutdown hooks for the async HTTP client and idle loop.
- [ ] Run `pytest tests/test_api.py tests/test_runtime.py -q`.

### Task 6: Docs And Verification

**Files:**
- Create: `README.md`
- Create: `.env.example`

- [ ] Document environment variables and local startup commands.
- [ ] Document Java integration settings required for联调.
- [ ] Run `pytest -q`.
- [ ] Run a minimal import check: `python -c "from deep_cover_agent.api import create_app; app = create_app(); print(app.title)"`.
