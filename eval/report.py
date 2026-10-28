from __future__ import annotations

import json
from pathlib import Path

from eval.scoring import SuiteScore

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def log_to_wandb(
    suite_score: SuiteScore,
    run_name: str,
    project: str = "agent-eval-harness",
    tags: list[str] | None = None,
    config: dict | None = None,
) -> str | None:
    if not HAS_WANDB:
        return None

    run = wandb.init(project=project, name=run_name, tags=tags or [], config=config or {})

    run.log({
        "success_rate": suite_score.success_rate,
        "mean_tool_accuracy": suite_score.mean_tool_accuracy,
        "mean_efficiency": suite_score.mean_efficiency,
        "total_cost_usd": suite_score.total_cost,
        "mean_cost_usd": suite_score.mean_cost,
        "escalation_rate": suite_score.escalation_rate,
        "num_scenarios": len(suite_score.scores),
    })

    columns = ["scenario_id", "task_success", "tool_accuracy", "efficiency", "cost_usd", "error"]
    table = wandb.Table(columns=columns)
    for s in suite_score.scores:
        table.add_data(s.scenario_id, s.task_success, s.tool_accuracy, s.efficiency, s.cost_usd, s.error or "")
    run.log({"scenario_results": table})

    run.finish()
    return run.url


def generate_markdown_report(suite_score: SuiteScore, output_path: Path | None = None) -> str:
    lines = [
        "# Agent Evaluation Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Success Rate | {suite_score.success_rate:.1%} |",
        f"| Mean Tool Accuracy | {suite_score.mean_tool_accuracy:.1%} |",
        f"| Mean Efficiency | {suite_score.mean_efficiency:.1%} |",
        f"| Total Cost | ${suite_score.total_cost:.4f} |",
        f"| Mean Cost/Task | ${suite_score.mean_cost:.4f} |",
        f"| Escalation Rate | {suite_score.escalation_rate:.1%} |",
        f"| Scenarios Run | {len(suite_score.scores)} |",
        "",
        "## Per-Scenario Results",
        "",
        "| Scenario | Success | Tool Acc. | Efficiency | Cost |",
        "|----------|---------|-----------|------------|------|",
    ]

    for s in suite_score.scores:
        status = "PASS" if s.task_success else "FAIL"
        lines.append(
            f"| {s.scenario_id} | {status} | {s.tool_accuracy:.1%} | {s.efficiency:.1%} | ${s.cost_usd:.4f} |"
        )

    if any(s.error for s in suite_score.scores):
        lines.extend(["", "## Errors", ""])
        for s in suite_score.scores:
            if s.error:
                lines.append(f"- **{s.scenario_id}**: {s.error}")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
