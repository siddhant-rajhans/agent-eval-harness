from pathlib import Path

from eval.scenarios import ExpectedToolCall, Scenario, ScenarioSuite
from eval.scoring import (
    ScenarioScore,
    SuiteScore,
    compute_efficiency,
    compute_tool_accuracy,
    estimate_cost,
)


class TestToolAccuracy:
    def test_perfect_match(self):
        actual = ["lookup_order", "process_refund"]
        expected = [ExpectedToolCall(name="lookup_order"), ExpectedToolCall(name="process_refund")]
        acc, prec, rec = compute_tool_accuracy(actual, expected)
        assert acc == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_extra_tool_call(self):
        actual = ["lookup_order", "process_refund", "check_inventory"]
        expected = [ExpectedToolCall(name="lookup_order"), ExpectedToolCall(name="process_refund")]
        acc, prec, rec = compute_tool_accuracy(actual, expected)
        assert rec == 1.0
        assert prec < 1.0

    def test_missing_tool_call(self):
        actual = ["lookup_order"]
        expected = [ExpectedToolCall(name="lookup_order"), ExpectedToolCall(name="process_refund")]
        acc, prec, rec = compute_tool_accuracy(actual, expected)
        assert rec == 0.5
        assert prec == 1.0

    def test_no_tools_expected_or_called(self):
        acc, prec, rec = compute_tool_accuracy([], [])
        assert acc == 1.0

    def test_completely_wrong_tools(self):
        actual = ["check_inventory"]
        expected = [ExpectedToolCall(name="process_refund")]
        acc, prec, rec = compute_tool_accuracy(actual, expected)
        assert acc == 0.0
        assert rec == 0.0


class TestEfficiency:
    def test_optimal(self):
        assert compute_efficiency(3, 3) == 1.0

    def test_under_optimal(self):
        assert compute_efficiency(2, 3) == 1.0

    def test_over_optimal(self):
        score = compute_efficiency(6, 3)
        assert 0.0 < score < 1.0


class TestCostEstimation:
    def test_gpt4o_mini_cost(self):
        cost = estimate_cost(1000, "gpt-4o-mini")
        assert cost == 0.00015

    def test_zero_tokens(self):
        assert estimate_cost(0) == 0.0


class TestSuiteScore:
    def test_success_rate(self):
        scores = [
            ScenarioScore("s1", 1, 0.9, 0.9, 0.9, 0.8, False, 2, 2, 0.001),
            ScenarioScore("s2", 0, 0.5, 0.5, 0.5, 0.6, False, 3, 3, 0.002),
            ScenarioScore("s3", 1, 1.0, 1.0, 1.0, 1.0, False, 1, 1, 0.001),
        ]
        suite = SuiteScore(scores=scores)
        assert abs(suite.success_rate - 2 / 3) < 1e-6

    def test_regression_check_detects_drop(self):
        baseline = SuiteScore(scores=[
            ScenarioScore("s1", 1, 0.9, 0.9, 0.9, 0.8, False, 2, 2, 0.001),
            ScenarioScore("s2", 1, 0.9, 0.9, 0.9, 0.8, False, 2, 2, 0.001),
        ])
        current = SuiteScore(scores=[
            ScenarioScore("s1", 0, 0.5, 0.5, 0.5, 0.6, False, 3, 3, 0.005),
            ScenarioScore("s2", 0, 0.5, 0.5, 0.5, 0.6, False, 3, 3, 0.005),
        ])
        regressions = current.regression_check(baseline)
        assert len(regressions) >= 1

    def test_no_regression(self):
        scores = [ScenarioScore("s1", 1, 0.9, 0.9, 0.9, 0.8, False, 2, 2, 0.001)]
        suite = SuiteScore(scores=scores)
        regressions = suite.regression_check(suite)
        assert len(regressions) == 0


class TestScenarioSuite:
    def test_load_from_json(self):
        suite = ScenarioSuite.from_json(Path("scenarios/customer_support.json"))
        assert len(suite.scenarios) == 50
        assert suite.scenarios[0].id == "CS-001"

    def test_filter_by_tag(self):
        suite = ScenarioSuite.from_json(Path("scenarios/customer_support.json"))
        refund_scenarios = suite.filter_by_tag("refund")
        assert len(refund_scenarios) > 0
        assert all("refund" in s.tags for s in refund_scenarios)

    def test_filter_by_difficulty(self):
        suite = ScenarioSuite.from_json(Path("scenarios/customer_support.json"))
        easy = suite.filter_by_difficulty("easy")
        assert len(easy) > 0
        assert all(s.difficulty == "easy" for s in easy)
