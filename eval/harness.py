from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.config import AgentConfig, EvalConfig
from agent.graph import build_agent
from agent.prompts import SYSTEM_PROMPT
from eval.scenarios import Scenario, ScenarioSuite
from eval.scoring import (
    ScenarioScore,
    SuiteScore,
    compute_efficiency,
    compute_tool_accuracy,
    estimate_cost,
)


@dataclass
class RunTrace:
    scenario_id: str
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    total_tokens: int = 0
    error: str | None = None


def _count_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += len(content) // 4
    return total


def run_scenario(scenario: Scenario, agent_config: AgentConfig) -> RunTrace:
    agent = build_agent(agent_config)
    trace = RunTrace(scenario_id=scenario.id)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    state = {
        "messages": messages,
        "tool_calls_made": [],
        "escalated": False,
        "turn_count": 0,
        "retry_count": 0,
    }

    start = time.monotonic()
    try:
        for user_msg in scenario.user_messages:
            state["messages"].append(HumanMessage(content=user_msg))
            trace.messages.append({"role": "user", "content": user_msg})

            result = agent.invoke(state)
            state = result

            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage):
                    trace.messages.append({"role": "assistant", "content": msg.content})
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            trace.tool_calls.append(tc["name"])
                            trace.messages.append({
                                "role": "tool_call",
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                            })

        trace.total_tokens = _count_tokens(state.get("messages", []))
    except Exception as e:
        trace.error = str(e)

    trace.latency_seconds = time.monotonic() - start
    trace.tool_calls = state.get("tool_calls_made", trace.tool_calls)
    return trace


def score_trace(trace: RunTrace, scenario: Scenario, model: str) -> ScenarioScore:
    if trace.error:
        return ScenarioScore(
            scenario_id=scenario.id,
            task_success=0,
            tool_accuracy=0.0,
            tool_precision=0.0,
            tool_recall=0.0,
            efficiency=0.0,
            escalated=False,
            turn_count=0,
            total_tool_calls=0,
            cost_usd=0.0,
            error=trace.error,
        )

    accuracy, precision, recall = compute_tool_accuracy(trace.tool_calls, scenario.expected_tools)
    expected_tool_names = {tc.name for tc in scenario.expected_tools}
    escalated = "escalate_to_human" in trace.tool_calls

    task_success = 1 if recall >= 0.8 else 0
    if scenario.expected_outcome == "escalation":
        task_success = 1 if escalated else 0
    elif scenario.expected_outcome == "refund_approved" and "process_refund" not in trace.tool_calls:
        task_success = 0

    user_turns = sum(1 for m in trace.messages if m.get("role") == "user")
    efficiency = compute_efficiency(user_turns + len(trace.tool_calls), len(scenario.expected_tools) + len(scenario.user_messages))

    return ScenarioScore(
        scenario_id=scenario.id,
        task_success=task_success,
        tool_accuracy=accuracy,
        tool_precision=precision,
        tool_recall=recall,
        efficiency=efficiency,
        escalated=escalated,
        turn_count=user_turns,
        total_tool_calls=len(trace.tool_calls),
        cost_usd=estimate_cost(trace.total_tokens, model),
    )


class EvalHarness:
    def __init__(self, agent_config: AgentConfig, eval_config: EvalConfig):
        self.agent_config = agent_config
        self.eval_config = eval_config
        self.traces: list[RunTrace] = []
        self.suite_score: SuiteScore | None = None

    def run_suite(self, suite: ScenarioSuite) -> SuiteScore:
        scores = []
        for scenario in suite.scenarios:
            trace = run_scenario(scenario, self.agent_config)
            self.traces.append(trace)
            score = score_trace(trace, scenario, self.agent_config.model)
            scores.append(score)

        self.suite_score = SuiteScore(scores=scores)
        return self.suite_score

    def save_results(self, output_dir: Path | None = None) -> Path:
        output_dir = output_dir or self.eval_config.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "summary": {
                "success_rate": self.suite_score.success_rate if self.suite_score else 0,
                "mean_tool_accuracy": self.suite_score.mean_tool_accuracy if self.suite_score else 0,
                "mean_efficiency": self.suite_score.mean_efficiency if self.suite_score else 0,
                "total_cost": self.suite_score.total_cost if self.suite_score else 0,
                "mean_cost": self.suite_score.mean_cost if self.suite_score else 0,
                "escalation_rate": self.suite_score.escalation_rate if self.suite_score else 0,
            },
            "scores": [
                {
                    "scenario_id": s.scenario_id,
                    "task_success": s.task_success,
                    "tool_accuracy": s.tool_accuracy,
                    "efficiency": s.efficiency,
                    "cost_usd": s.cost_usd,
                    "error": s.error,
                }
                for s in (self.suite_score.scores if self.suite_score else [])
            ],
            "traces": [
                {
                    "scenario_id": t.scenario_id,
                    "messages": t.messages,
                    "tool_calls": t.tool_calls,
                    "latency_seconds": t.latency_seconds,
                    "total_tokens": t.total_tokens,
                }
                for t in self.traces
            ],
        }

        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results_path
