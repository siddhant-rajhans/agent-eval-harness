from __future__ import annotations

import logging
from dataclasses import dataclass, field

from eval.scenarios import ExpectedToolCall

logger = logging.getLogger(__name__)

# Offline cost estimation rates ($/1K tokens). Used for relative comparison
# between model choices, not billing-accurate accounting.
MODEL_COST_PER_1K: dict[str, float] = {
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.005,
    "claude-sonnet-4-20250514": 0.003,
}
_DEFAULT_COST_PER_1K = 0.001


@dataclass
class ScenarioScore:
    """Per-scenario evaluation: task success, tool accuracy, efficiency, and cost."""
    scenario_id: str
    task_success: int
    tool_accuracy: float
    tool_precision: float
    tool_recall: float
    efficiency: float
    escalated: bool
    turn_count: int
    total_tool_calls: int
    cost_usd: float
    error: str | None = None

    @property
    def composite_score(self) -> float:
        # Weights: task_success 40%, tool_accuracy 30%, efficiency 30%
        return (
            self.task_success * 0.4
            + self.tool_accuracy * 0.3
            + self.efficiency * 0.3
        )


@dataclass
class SuiteScore:
    """Aggregate scores across all scenarios in an evaluation run."""
    scores: list[ScenarioScore] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.task_success for s in self.scores) / len(self.scores)

    @property
    def mean_tool_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.tool_accuracy for s in self.scores) / len(self.scores)

    @property
    def mean_efficiency(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.efficiency for s in self.scores) / len(self.scores)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.scores)

    @property
    def mean_cost(self) -> float:
        if not self.scores:
            return 0.0
        return self.total_cost / len(self.scores)

    @property
    def escalation_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.escalated) / len(self.scores)

    def regression_check(self, baseline: SuiteScore, threshold: float = 0.05) -> list[str]:
        regressions = []
        if self.success_rate < baseline.success_rate - threshold:
            regressions.append(
                f"Success rate dropped: {baseline.success_rate:.1%} -> {self.success_rate:.1%}"
            )
        if self.mean_tool_accuracy < baseline.mean_tool_accuracy - threshold:
            regressions.append(
                f"Tool accuracy dropped: {baseline.mean_tool_accuracy:.1%} -> {self.mean_tool_accuracy:.1%}"
            )
        if self.mean_cost > baseline.mean_cost * 1.2:
            regressions.append(
                f"Cost increased: ${baseline.mean_cost:.4f} -> ${self.mean_cost:.4f}"
            )
        return regressions


def compute_tool_accuracy(
    actual_calls: list[str],
    expected: list[ExpectedToolCall],
) -> tuple[float, float, float]:
    expected_names = {tc.name for tc in expected}
    actual_set = set(actual_calls)

    if not expected_names and not actual_set:
        return 1.0, 1.0, 1.0

    true_positives = expected_names & actual_set
    precision = len(true_positives) / len(actual_set) if actual_set else 0.0
    recall = len(true_positives) / len(expected_names) if expected_names else 1.0
    accuracy = len(true_positives) / len(expected_names | actual_set) if (expected_names | actual_set) else 1.0

    return accuracy, precision, recall


def compute_efficiency(turn_count: int, optimal_turns: int | None = None) -> float:
    if optimal_turns is None:
        optimal_turns = max(1, turn_count // 2)
    if turn_count <= optimal_turns:
        return 1.0
    return max(0.0, 1.0 - (turn_count - optimal_turns) / (optimal_turns * 2))


def estimate_cost(total_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estimate cost from token count. Uses len(content)//4 token approximation
    (standard for offline eval without a tiktoken dependency)."""
    rate = MODEL_COST_PER_1K.get(model, _DEFAULT_COST_PER_1K)
    cost = (total_tokens / 1000) * rate
    logger.debug("Cost estimate: %d tokens @ %s = $%.6f", total_tokens, model, cost)
    return cost
