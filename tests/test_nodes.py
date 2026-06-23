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


async def test_image_query_reports_service_error_when_vision_fails(monkeypatch, tmp_path):
    """A vision-service failure should say 'try later', not 'reupload' — the
    image was fine, the problem is server-side."""
    img = tmp_path / "pic.jpg"
    img.write_bytes(b"fake")  # only needs to exist; _encode_image is mocked

    monkeypatch.setattr(builder.settings, "VISION_API_KEY", "k")
    monkeypatch.setattr(builder.settings, "VISION_BASE_URL", "http://x")
    monkeypatch.setattr(builder.settings, "VISION_MODEL", "m")
    monkeypatch.setattr(builder, "_encode_image", lambda path: "ZmFrZQ==")

    async def _fail(image_data):
        raise RuntimeError("vision down")

    monkeypatch.setattr(builder, "_describe_image", _fail)

    state = AgentState(
        messages=[HumanMessage(content="看看这个")],
        router={"type": "image-query", "logic": ""},
    )
    result = await builder.create_image_query(
        state, config={"configurable": {"image_path": str(img)}}
    )
    assert "暂时不可用" in result["messages"][0].content


async def test_image_query_happy_path_feeds_description_into_reply(monkeypatch, tmp_path):
    img = tmp_path / "pic.jpg"
    img.write_bytes(b"fake")

    monkeypatch.setattr(builder.settings, "VISION_API_KEY", "k")
    monkeypatch.setattr(builder.settings, "VISION_BASE_URL", "http://x")
    monkeypatch.setattr(builder.settings, "VISION_MODEL", "m")
    monkeypatch.setattr(builder, "_encode_image", lambda path: "ZmFrZQ==")

    async def _describe(image_data):
        return "一把智能门锁"

    monkeypatch.setattr(builder, "_describe_image", _describe)

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=AIMessage(content="亲~这是智能门锁哦😊"))
    monkeypatch.setattr(builder.LLMFactory, "create_agent_model", lambda **kwargs: fake_model)

    state = AgentState(
        messages=[HumanMessage(content="这是啥")],
        router={"type": "image-query", "logic": ""},
    )
    result = await builder.create_image_query(
        state, config={"configurable": {"image_path": str(img)}}
    )

    assert result["messages"][0].content == "亲~这是智能门锁哦😊"
    # the vision description was injected into the reply's system prompt
    sent = fake_model.ainvoke.call_args.args[0]
    assert "一把智能门锁" in sent[0]["content"]


async def test_file_query_without_file_asks_to_upload():
    state = AgentState(
        messages=[HumanMessage(content="看看我的文件")],
        router={"type": "file-query", "logic": ""},
    )
    result = await builder.create_file_query(state, config={"configurable": {}})
    assert "上传文件" in result["messages"][0].content


async def test_file_query_no_results_says_not_found(monkeypatch, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")  # 只需存在;query_file 被 mock
    fake = MagicMock()
    fake.query_file = AsyncMock(return_value=[])
    monkeypatch.setattr(builder, "_get_embedding_service", lambda: fake)
    state = AgentState(
        messages=[HumanMessage(content="保修多久")],
        router={"type": "file-query", "logic": ""},
    )
    result = await builder.create_file_query(
        state, config={"configurable": {"file_path": str(f)}}
    )
    assert "没有找到相关信息" in result["messages"][0].content


async def test_file_query_unreadable_file_reports_error(monkeypatch, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")

    async def _raise(*args, **kwargs):
        raise RuntimeError("bad pdf")

    fake = MagicMock()
    fake.query_file = _raise
    monkeypatch.setattr(builder, "_get_embedding_service", lambda: fake)
    state = AgentState(
        messages=[HumanMessage(content="保修多久")],
        router={"type": "file-query", "logic": ""},
    )
    result = await builder.create_file_query(
        state, config={"configurable": {"file_path": str(f)}}
    )
    assert "无法读取" in result["messages"][0].content


async def test_file_query_happy_path_feeds_retrieved_context(monkeypatch, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    fake = MagicMock()
    fake.query_file = AsyncMock(return_value=[{"content": "保修期是两年"}])
    monkeypatch.setattr(builder, "_get_embedding_service", lambda: fake)
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=AIMessage(content="亲~保修两年哦😊"))
    monkeypatch.setattr(
        builder.LLMFactory, "create_agent_model", lambda **kwargs: fake_model
    )
    state = AgentState(
        messages=[HumanMessage(content="保修多久")],
        router={"type": "file-query", "logic": ""},
    )
    result = await builder.create_file_query(
        state, config={"configurable": {"file_path": str(f)}}
    )
    assert result["messages"][0].content == "亲~保修两年哦😊"
    sent = fake_model.ainvoke.call_args.args[0]
    assert "保修期是两年" in sent[0]["content"]  # 检索片段进了 prompt


def _router_model_returning(router_type: str):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value={
            "type": router_type,
            "logic": "x",
            "confidence": "high",
            "question": "",
        }
    )
    fake_model = MagicMock()
    fake_model.with_structured_output = MagicMock(return_value=structured)
    return fake_model


async def test_uploaded_file_forces_file_query_routing(monkeypatch):
    # 上传文件是强信号 -> 即使 LLM 判成别的,也强制走 file-query
    monkeypatch.setattr(
        builder.LLMFactory,
        "create_agent_model",
        lambda **kwargs: _router_model_returning("general-query"),
    )
    state = AgentState(messages=[HumanMessage(content="看看这个")])
    result = await builder.analyze_and_route_query(
        state, config={"configurable": {"file_path": "/x.pdf"}}
    )
    assert result["router"]["type"] == "file-query"


async def test_uploaded_image_forces_image_query_routing(monkeypatch):
    monkeypatch.setattr(
        builder.LLMFactory,
        "create_agent_model",
        lambda **kwargs: _router_model_returning("general-query"),
    )
    state = AgentState(messages=[HumanMessage(content="看看这个")])
    result = await builder.analyze_and_route_query(
        state, config={"configurable": {"image_path": "/x.jpg"}}
    )
    assert result["router"]["type"] == "image-query"
