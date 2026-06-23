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
