import os

import pytest

from agent.config import AgentConfig
from agent.graph import GraphState, build_agent


class TestGraphBuild:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No API key")
    def test_agent_compiles(self):
        agent = build_agent(AgentConfig())
        assert agent is not None

    def test_graph_state_defaults(self):
        state = GraphState()
        assert state.messages == []
        assert state.tool_calls_made == []
        assert state.escalated is False
        assert state.turn_count == 0

    def test_graph_state_tracks_tools(self):
        state = GraphState(tool_calls_made=["lookup_order", "process_refund"])
        assert len(state.tool_calls_made) == 2
