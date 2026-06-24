"""Tests for the file-query grounding gate (output validation / hallucination check).

The judge LLM is faked, so these are deterministic and cost no tokens. We only
verify the wiring: a grounded answer passes through, an ungrounded one is
replaced by the safe fallback, and a judge error fails open (keeps the answer).
"""
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

import app.lg_agent.builder as builder
from app.lg_agent.state import AgentState


def _patch_models(monkeypatch, *, answer: str, grounded: bool):
    """Fake both LLMs create_file_query uses: the generator and the grounding judge.

    Dispatch by tags so the same factory call site returns the right fake.
    """
    generator = MagicMock()
    generator.ainvoke = AsyncMock(return_value=AIMessage(content=answer))

    verdict_runnable = MagicMock()
    verdict_runnable.ainvoke = AsyncMock(
        return_value=builder.GroundingVerdict(grounded=grounded)
    )
    judge = MagicMock()
    judge.with_structured_output = MagicMock(return_value=verdict_runnable)

    def _create(**kwargs):
        return judge if kwargs.get("tags") == ["grounding_check"] else generator

    monkeypatch.setattr(builder.LLMFactory, "create_agent_model", _create)


def _patch_retrieval(monkeypatch):
    fake = MagicMock()
    fake.query_file = AsyncMock(return_value=[{"content": "保修期为 2 年"}])
    monkeypatch.setattr(builder, "_get_embedding_service", lambda: fake)


async def test_grounded_answer_passes_through(monkeypatch, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    _patch_retrieval(monkeypatch)
    _patch_models(monkeypatch, answer="保修期是 2 年", grounded=True)

    state = AgentState(
        messages=[HumanMessage(content="保修多久")],
        router={"type": "file-query", "logic": ""},
    )
    result = await builder.create_file_query(
        state, config={"configurable": {"file_path": str(f)}}
    )
    assert result["messages"][0].content == "保修期是 2 年"


async def test_ungrounded_answer_replaced_with_fallback(monkeypatch, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    _patch_retrieval(monkeypatch)
    _patch_models(monkeypatch, answer="保修期是 10 年还送终身保养", grounded=False)

    state = AgentState(
        messages=[HumanMessage(content="保修多久")],
        router={"type": "file-query", "logic": ""},
    )
    result = await builder.create_file_query(
        state, config={"configurable": {"file_path": str(f)}}
    )
    content = result["messages"][0].content
    assert content == builder.FILE_UNGROUNDED_MSG
    assert "10 年" not in content  # the hallucinated answer must not leak through


async def test_verify_grounding_fails_open_on_judge_error(monkeypatch):
    """A judge infra error must not block legit answers — fail open (treat as grounded)."""
    judge = MagicMock()
    failing = MagicMock()
    failing.ainvoke = AsyncMock(side_effect=RuntimeError("judge down"))
    judge.with_structured_output = MagicMock(return_value=failing)
    monkeypatch.setattr(
        builder.LLMFactory, "create_agent_model", lambda **kwargs: judge
    )

    assert await builder._verify_grounding("some context", "some answer") is True
