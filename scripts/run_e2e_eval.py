"""End-to-end evaluation of the scoring pipeline against all 50 scenarios.

This exercises the scoring, regression detection, and report generation
WITHOUT calling an LLM. It creates mock tool-call traces (perfect, degraded,
and wrong-tool variants) and pushes them through the same code path the
live harness uses, so every number printed below is a real measurement of
the scoring logic.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from eval.scenarios import ScenarioSuite
from eval.scoring import (
    ScenarioScore,
    SuiteScore,
    compute_efficiency,
    compute_tool_accuracy,
    estimate_cost,
)
from eval.report import generate_markdown_report

SCENARIOS_PATH = Path("scenarios/customer_support.json")

random.seed(42)


# ---------------------------------------------------------------------------
# Helpers to synthesise mock traces
# ---------------------------------------------------------------------------

def perfect_trace(scenario) -> list[str]:
    """Return exactly the expected tool calls."""
    return [tc.name for tc in scenario.expected_tools]


def degraded_trace(scenario) -> list[str]:
    """Drop one expected tool (if any) and add one spurious call."""
    calls = perfect_trace(scenario)
    if calls:
        calls = calls[:-1]  # miss the last expected tool
    calls.append("check_inventory")  # spurious extra
    return calls


def wrong_trace(scenario) -> list[str]:
    """Completely wrong tool calls."""
    return ["escalate_to_human", "update_shipping_address"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_from_trace(scenario, tool_calls: list[str], label: str) -> ScenarioScore:
    accuracy, precision, recall = compute_tool_accuracy(tool_calls, scenario.expected_tools)
    escalated = "escalate_to_human" in tool_calls

    # Task success: same logic as harness.py
    task_success = 1 if recall >= 0.8 else 0
    if scenario.expected_outcome == "escalation":
        task_success = 1 if escalated else 0
    elif scenario.expected_outcome == "refund_approved" and "process_refund" not in tool_calls:
        task_success = 0

    user_turns = len(scenario.user_messages)
    optimal = len(scenario.expected_tools) + user_turns
    efficiency = compute_efficiency(user_turns + len(tool_calls), optimal)

    # Estimate token cost from a realistic per-turn count
    est_tokens = user_turns * 250 + len(tool_calls) * 350
    cost = estimate_cost(est_tokens, "gpt-4o-mini")

    return ScenarioScore(
        scenario_id=f"{scenario.id}-{label}",
        task_success=task_success,
        tool_accuracy=accuracy,
        tool_precision=precision,
        tool_recall=recall,
        efficiency=efficiency,
        escalated=escalated,
        turn_count=user_turns,
        total_tool_calls=len(tool_calls),
        cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    suite = ScenarioSuite.from_json(SCENARIOS_PATH)
    print(f"Loaded {len(suite.scenarios)} scenarios from {SCENARIOS_PATH}\n")

    # ---- 1. Perfect-trace evaluation (baseline) ---
    print("=" * 60)
    print("1. PERFECT-TRACE EVALUATION (baseline)")
    print("=" * 60)
    t0 = time.perf_counter()
    perfect_scores: list[ScenarioScore] = []
    for s in suite.scenarios:
        trace = perfect_trace(s)
        score = score_from_trace(s, trace, "perfect")
        perfect_scores.append(score)
    baseline = SuiteScore(scores=perfect_scores)
    elapsed_perfect = time.perf_counter() - t0

    print(f"  Scenarios evaluated : {len(baseline.scores)}")
    print(f"  Success rate        : {baseline.success_rate:.1%}")
    print(f"  Mean tool accuracy  : {baseline.mean_tool_accuracy:.1%}")
    print(f"  Mean efficiency     : {baseline.mean_efficiency:.1%}")
    print(f"  Total cost (est.)   : ${baseline.total_cost:.4f}")
    print(f"  Mean cost/task      : ${baseline.mean_cost:.4f}")
    print(f"  Escalation rate     : {baseline.escalation_rate:.1%}")
    print(f"  Scoring time        : {elapsed_perfect*1000:.1f} ms")

    # List any imperfect scores
    failed = [s for s in perfect_scores if s.task_success == 0]
    if failed:
        print(f"\n  *** {len(failed)} scenario(s) scored as failed even with perfect traces:")
        for f in failed:
            print(f"      {f.scenario_id}: accuracy={f.tool_accuracy:.2f} recall={f.tool_recall:.2f}")

    # ---- 2. Degraded-trace evaluation ---
    print(f"\n{'=' * 60}")
    print("2. DEGRADED-TRACE EVALUATION (missing + spurious tools)")
    print("=" * 60)
    t0 = time.perf_counter()
    degraded_scores: list[ScenarioScore] = []
    for s in suite.scenarios:
        trace = degraded_trace(s)
        score = score_from_trace(s, trace, "degraded")
        degraded_scores.append(score)
    degraded = SuiteScore(scores=degraded_scores)
    elapsed_degraded = time.perf_counter() - t0

    print(f"  Success rate        : {degraded.success_rate:.1%}")
    print(f"  Mean tool accuracy  : {degraded.mean_tool_accuracy:.1%}")
    print(f"  Mean efficiency     : {degraded.mean_efficiency:.1%}")
    print(f"  Mean cost/task      : ${degraded.mean_cost:.4f}")
    print(f"  Scoring time        : {elapsed_degraded*1000:.1f} ms")

    # ---- 3. Wrong-trace evaluation ---
    print(f"\n{'=' * 60}")
    print("3. WRONG-TRACE EVALUATION (completely wrong tools)")
    print("=" * 60)
    t0 = time.perf_counter()
    wrong_scores: list[ScenarioScore] = []
    for s in suite.scenarios:
        trace = wrong_trace(s)
        score = score_from_trace(s, trace, "wrong")
        wrong_scores.append(score)
    wrong = SuiteScore(scores=wrong_scores)
    elapsed_wrong = time.perf_counter() - t0

    print(f"  Success rate        : {wrong.success_rate:.1%}")
    print(f"  Mean tool accuracy  : {wrong.mean_tool_accuracy:.1%}")
    print(f"  Scoring time        : {elapsed_wrong*1000:.1f} ms")

    # ---- 4. Regression detection ---
    print(f"\n{'=' * 60}")
    print("4. REGRESSION DETECTION")
    print("=" * 60)
    regressions_degraded = degraded.regression_check(baseline, threshold=0.05)
    regressions_wrong = wrong.regression_check(baseline, threshold=0.05)

    print(f"  Baseline vs degraded : {len(regressions_degraded)} regression(s)")
    for r in regressions_degraded:
        print(f"    - {r}")

    print(f"  Baseline vs wrong    : {len(regressions_wrong)} regression(s)")
    for r in regressions_wrong:
        print(f"    - {r}")

    if not regressions_degraded and not regressions_wrong:
        print("  *** WARNING: no regressions detected -- check thresholds")

    # ---- 5. Markdown report generation ---
    print(f"\n{'=' * 60}")
    print("5. REPORT GENERATION")
    print("=" * 60)
    report_path = Path("results/baseline_report.md")
    report = generate_markdown_report(baseline, report_path)
    print(f"  Report written to {report_path} ({len(report)} chars)")
    print(f"  Report contains {report.count('|')} table cells")

    # ---- 6. Cost estimation sanity checks ---
    print(f"\n{'=' * 60}")
    print("6. COST ESTIMATION")
    print("=" * 60)
    models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]
    for m in models:
        c = estimate_cost(10000, m)
        print(f"  10K tokens @ {m:30s} = ${c:.4f}")

    # ---- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total scenarios        : {len(suite.scenarios)}")
    print(f"  Baseline success rate  : {baseline.success_rate:.1%}")
    print(f"  Baseline tool accuracy : {baseline.mean_tool_accuracy:.1%}")
    print(f"  Baseline efficiency    : {baseline.mean_efficiency:.1%}")
    print(f"  Baseline mean cost     : ${baseline.mean_cost:.6f}")
    print(f"  Regressions caught     : {len(regressions_degraded) + len(regressions_wrong)}")
    print(f"  Total scoring time     : {(elapsed_perfect+elapsed_degraded+elapsed_wrong)*1000:.1f} ms")

    # Save machine-readable results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary = {
        "num_scenarios": len(suite.scenarios),
        "baseline": {
            "success_rate": baseline.success_rate,
            "mean_tool_accuracy": baseline.mean_tool_accuracy,
            "mean_efficiency": baseline.mean_efficiency,
            "total_cost": baseline.total_cost,
            "mean_cost": baseline.mean_cost,
            "escalation_rate": baseline.escalation_rate,
        },
        "degraded": {
            "success_rate": degraded.success_rate,
            "mean_tool_accuracy": degraded.mean_tool_accuracy,
        },
        "wrong": {
            "success_rate": wrong.success_rate,
            "mean_tool_accuracy": wrong.mean_tool_accuracy,
        },
        "regressions_detected": len(regressions_degraded) + len(regressions_wrong),
    }
    (results_dir / "e2e_eval_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to results/e2e_eval_results.json")


if __name__ == "__main__":
    main()
