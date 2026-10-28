from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agent.config import AgentConfig, EvalConfig
from eval.harness import EvalHarness
from eval.report import generate_markdown_report, log_to_wandb
from eval.scenarios import ScenarioSuite

console = Console()


@click.group()
def cli():
    """Agent evaluation harness for customer support workflows."""


@cli.command()
@click.option("--scenarios", "-s", type=click.Path(exists=True), required=True, help="Path to scenarios JSON file")
@click.option("--model", "-m", default="gpt-4o-mini", help="LLM model to use")
@click.option("--output", "-o", type=click.Path(), default="results", help="Output directory")
@click.option("--wandb/--no-wandb", default=False, help="Log results to W&B")
@click.option("--tag", multiple=True, help="Filter scenarios by tag")
@click.option("--run-name", default=None, help="W&B run name")
def run(scenarios: str, model: str, output: str, wandb: bool, tag: tuple[str, ...], run_name: str | None):
    """Run the evaluation harness against a scenario suite."""
    agent_config = AgentConfig(model=model)
    eval_config = EvalConfig(results_dir=Path(output), wandb_enabled=wandb)

    suite = ScenarioSuite.from_json(Path(scenarios))
    if tag:
        filtered = []
        for t in tag:
            filtered.extend(suite.filter_by_tag(t))
        seen = set()
        suite.scenarios = [s for s in filtered if s.id not in seen and not seen.add(s.id)]

    console.print(f"[bold]Running {len(suite.scenarios)} scenarios with {model}[/bold]")

    harness = EvalHarness(agent_config, eval_config)
    suite_score = harness.run_suite(suite)

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Success Rate", f"{suite_score.success_rate:.1%}")
    table.add_row("Tool Accuracy", f"{suite_score.mean_tool_accuracy:.1%}")
    table.add_row("Efficiency", f"{suite_score.mean_efficiency:.1%}")
    table.add_row("Mean Cost/Task", f"${suite_score.mean_cost:.4f}")
    table.add_row("Total Cost", f"${suite_score.total_cost:.4f}")
    table.add_row("Escalation Rate", f"{suite_score.escalation_rate:.1%}")
    console.print(table)

    results_path = harness.save_results()
    console.print(f"\nResults saved to {results_path}")

    report_path = Path(output) / "report.md"
    generate_markdown_report(suite_score, report_path)
    console.print(f"Report saved to {report_path}")

    if wandb:
        url = log_to_wandb(
            suite_score,
            run_name=run_name or f"eval-{model}",
            config={"model": model, "num_scenarios": len(suite.scenarios)},
        )
        if url:
            console.print(f"W&B run: {url}")


@cli.command()
@click.argument("results_a", type=click.Path(exists=True))
@click.argument("results_b", type=click.Path(exists=True))
@click.option("--threshold", "-t", type=float, default=0.05, help="Regression threshold")
def compare(results_a: str, results_b: str, threshold: float):
    """Compare two evaluation results and check for regressions."""
    import json

    with open(results_a) as f:
        data_a = json.load(f)
    with open(results_b) as f:
        data_b = json.load(f)

    console.print("[bold]Comparison: A (baseline) vs B (current)[/bold]\n")
    table = Table()
    table.add_column("Metric")
    table.add_column("Baseline (A)")
    table.add_column("Current (B)")
    table.add_column("Delta")

    for key in ["success_rate", "mean_tool_accuracy", "mean_efficiency", "mean_cost"]:
        val_a = data_a["summary"].get(key, 0)
        val_b = data_b["summary"].get(key, 0)
        delta = val_b - val_a
        sign = "+" if delta >= 0 else ""
        if key == "mean_cost":
            table.add_row(key, f"${val_a:.4f}", f"${val_b:.4f}", f"{sign}${delta:.4f}")
        else:
            table.add_row(key, f"{val_a:.1%}", f"{val_b:.1%}", f"{sign}{delta:.1%}")

    console.print(table)

    regressions = []
    sa, sb = data_a["summary"], data_b["summary"]
    if sb.get("success_rate", 0) < sa.get("success_rate", 0) - threshold:
        regressions.append("Success rate regression")
    if sb.get("mean_tool_accuracy", 0) < sa.get("mean_tool_accuracy", 0) - threshold:
        regressions.append("Tool accuracy regression")
    if sb.get("mean_cost", 0) > sa.get("mean_cost", 0) * 1.2:
        regressions.append("Cost regression (>20% increase)")

    if regressions:
        console.print(f"\n[bold red]REGRESSIONS DETECTED:[/bold red]")
        for r in regressions:
            console.print(f"  - {r}")
        raise SystemExit(1)
    else:
        console.print(f"\n[bold green]No regressions detected.[/bold green]")


if __name__ == "__main__":
    cli()
