"""Node-level tests with a faked LLM — deterministic, no network, no services.

These exercise the node logic (prompt assembly, return shape, fallback paths)
without calling DeepSeek/Ollama. Because model creation is centralized in
LLMFactory.create_agent_model, we fake the model in one place.
"""
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import app.lg_agent.builder as builder
from app.lg_agent.state import AgentState


async def test_general_query_assembles_prompt_and_returns_reply(monkeypatch):
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=AIMessage(content="亲～你好呀😊"))
    monkeypatch.setattr(
        builder.LLMFactory, "create_agent_model", lambda **kwargs: fake_model
    )

    state = AgentState(
        messages=[HumanMessage(content="你好")],
        router={"type": "general-query", "logic": "用户在闲聊问候"},
    )
    result = await builder.respond_to_general_query(state, config={})

    # return shape: the AI reply is appended under "messages"
    assert result["messages"][0].content == "亲～你好呀😊"

    # the model received the system prompt first, with the router logic injected,
    # followed by the conversation history
    sent = fake_model.ainvoke.call_args.args[0]
    assert sent[0]["role"] == "system"
    assert "用户在闲聊问候" in sent[0]["content"]
    assert sent[-1].content == "你好"


async def test_image_query_without_image_returns_friendly_message():
    state = AgentState(
        messages=[HumanMessage(content="帮我看看这个")],
        router={"type": "image-query", "logic": ""},
    )
    # no image_path in config -> node should short-circuit with a friendly message,
    # never touching the vision API
    result = await builder.create_image_query(state, config={"configurable": {}})

    assert "无法查看这张图片" in result["messages"][0].content


async def test_additional_info_survives_neo4j_being_down(monkeypatch):
    """Regression: if Neo4j is unavailable, get_additional_info must not crash
    with a NameError on an unbound neo4j_graph — it should still run guardrails.
    """
    def _neo4j_down():
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(builder, "get_neo4j_graph", _neo4j_down)

    # the guardrails chain is `prompt | model.with_structured_output(...)`;
    # make that structured runnable decide "end" (out of scope)
    fake_model = MagicMock()
    fake_model.with_structured_output.return_value = RunnableLambda(
        lambda _: builder.AdditionalGuardrailsOutput(decision="end")
    )
    monkeypatch.setattr(
        builder.LLMFactory, "create_agent_model", lambda **kwargs: fake_model
    )

    state = AgentState(
        messages=[HumanMessage(content="你们卖鞋吗")],
        router={"type": "additional-query", "logic": ""},
    )
    result = await builder.get_additional_info(state, config={})

    assert "没有这方面的商品" in result["messages"][0].content
