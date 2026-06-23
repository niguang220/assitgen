"""Unit tests for route_query — the router's branch logic.

Pure function: maps the classified query type to the next node name.
No LLM call, no database, no network.
"""
import pytest

from app.lg_agent.builder import route_query
from app.lg_agent.state import AgentState


def _state_with_type(router_type: str) -> AgentState:
    return AgentState(messages=[], router={"type": router_type, "logic": ""})


@pytest.mark.parametrize(
    "router_type, expected_node",
    [
        ("general-query", "respond_to_general_query"),
        ("additional-query", "get_additional_info"),
        ("graphrag-query", "create_research_plan"),
        ("image-query", "create_image_query"),
        ("file-query", "create_file_query"),
    ],
)
def test_each_type_routes_to_its_node(router_type, expected_node):
    assert route_query(_state_with_type(router_type)) == expected_node


def test_unknown_type_raises_value_error():
    with pytest.raises(ValueError):
        route_query(_state_with_type("nonsense-query"))


def _state(router_type: str, confidence: str) -> AgentState:
    return AgentState(
        messages=[], router={"type": router_type, "logic": "", "confidence": confidence}
    )


def test_low_confidence_graphrag_falls_back_to_clarification():
    # unsure + leaning toward a KB query -> ask for info instead of risking a wrong answer
    assert route_query(_state("graphrag-query", "low")) == "get_additional_info"


def test_high_confidence_graphrag_routes_normally():
    assert route_query(_state("graphrag-query", "high")) == "create_research_plan"


def test_low_confidence_does_not_override_chitchat_or_media():
    # low confidence on general/image/file still trusts the classified type
    assert route_query(_state("general-query", "low")) == "respond_to_general_query"
    assert route_query(_state("image-query", "low")) == "create_image_query"


def test_missing_confidence_routes_normally():
    # backward compatible: no confidence field -> behave as before
    assert route_query(_state_with_type("graphrag-query")) == "create_research_plan"
