from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProtocolConfig:
    version: str
    roles: list[str]
    turn_order: list[str]
    max_tokens: dict[str, int]
    critique: dict[str, Any]
    retry: dict[str, int]
    seed_policy: dict[str, int]

    @classmethod
    def from_file(cls, path: Path) -> "ProtocolConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=payload["version"],
            roles=payload["roles"],
            turn_order=payload["turn_order"],
            max_tokens=payload["max_tokens"],
            critique=payload["critique"],
            retry=payload["retry"],
            seed_policy=payload["seed_policy"],
        )


@dataclass
class StageEvent:
    stage: str
    role: str
    latency_ms: int
    token_count: int
    retries: int = 0
    accepted: bool | None = None
    errors: list[str] | None = None


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    seed: int
    user_query: str
    captain_plan: str
    role_outputs: dict[str, str]
    critiques: dict[str, dict[str, str]]
    final_answer: str
    events: list[StageEvent]
    metrics: dict[str, float]

