from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExpectedToolCall:
    name: str
    args: dict | None = None


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    user_messages: list[str]
    expected_tools: list[ExpectedToolCall]
    expected_outcome: str
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        expected_tools = [
            ExpectedToolCall(**tc) if isinstance(tc, dict) else ExpectedToolCall(name=tc)
            for tc in data.get("expected_tools", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            user_messages=data["user_messages"],
            expected_tools=expected_tools,
            expected_outcome=data["expected_outcome"],
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
        )


@dataclass
class ScenarioSuite:
    name: str
    scenarios: list[Scenario]

    @classmethod
    def from_json(cls, path: Path) -> ScenarioSuite:
        with open(path) as f:
            data = json.load(f)
        scenarios = [Scenario.from_dict(s) for s in data["scenarios"]]
        return cls(name=data.get("name", path.stem), scenarios=scenarios)

    def filter_by_tag(self, tag: str) -> list[Scenario]:
        return [s for s in self.scenarios if tag in s.tags]

    def filter_by_difficulty(self, difficulty: str) -> list[Scenario]:
        return [s for s in self.scenarios if s.difficulty == difficulty]
