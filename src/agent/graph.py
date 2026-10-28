from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agent.config import AgentConfig
from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS


class GraphState(BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    tool_calls_made: list[str] = Field(default_factory=list)
    escalated: bool = False
    turn_count: int = 0
    retry_count: int = 0

    model_config = {"arbitrary_types_allowed": True}


def build_agent(config: AgentConfig | None = None) -> StateGraph:
    config = config or AgentConfig()
    llm = ChatOpenAI(model=config.model, temperature=config.temperature)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: GraphState) -> dict[str, Any]:
        messages = state.messages
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        tool_names = [tc["name"] for tc in (response.tool_calls or [])]

        return {
            "messages": [response],
            "tool_calls_made": state.tool_calls_made + tool_names,
            "turn_count": state.turn_count + 1,
        }

    def should_continue(state: GraphState) -> Literal["tools", "end"]:
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            return "end"
        if last_message.tool_calls:
            return "tools"
        return "end"

    def check_escalation(state: GraphState) -> dict[str, Any]:
        if "escalate_to_human" in state.tool_calls_made:
            return {"escalated": True}
        return {}

    def should_retry_or_end(state: GraphState) -> Literal["agent", "end"]:
        if state.escalated:
            return "end"
        if state.turn_count >= (config.max_turns):
            return "end"
        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and not last.tool_calls:
            return "end"
        return "agent"

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_escalation", check_escalation)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "check_escalation")
    graph.add_conditional_edges(
        "check_escalation",
        should_retry_or_end,
        {"agent": "agent", "end": END},
    )

    return graph.compile()


def run_conversation(user_messages: list[str], config: AgentConfig | None = None) -> GraphState:
    agent = build_agent(config)
    state = GraphState(messages=[SystemMessage(content=SYSTEM_PROMPT)])

    for user_msg in user_messages:
        state.messages.append(HumanMessage(content=user_msg))
        result = agent.invoke({"messages": state.messages, **state.model_dump(exclude={"messages"})})
        state = GraphState(**result)

    return state
