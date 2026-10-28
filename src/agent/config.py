from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_turns: int = 15
    max_retries: int = 3
    retry_delay: float = 1.0
    escalation_threshold: int = 3

    @classmethod
    def from_yaml(cls, path: Path) -> AgentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.get("agent", {}).items() if k in cls.__dataclass_fields__})


@dataclass
class EvalConfig:
    scenarios_dir: Path = field(default_factory=lambda: Path("scenarios"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    wandb_project: str = "agent-eval-harness"
    wandb_enabled: bool = True
    parallel_workers: int = 4
    timeout_per_scenario: float = 120.0

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        raw = data.get("eval", {})
        if "scenarios_dir" in raw:
            raw["scenarios_dir"] = Path(raw["scenarios_dir"])
        if "results_dir" in raw:
            raw["results_dir"] = Path(raw["results_dir"])
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
