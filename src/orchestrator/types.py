from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProtocolConfig:
    version: str
    roles: list[str]
    turn_order: list[str]
    max_tokens: dict[str, int]
    harper: dict[str, Any]
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
            harper=payload.get("harper", {}),
            critique=payload["critique"],
            retry=payload["retry"],
            seed_policy=payload["seed_policy"],
        )


@dataclass
class RuntimeConfig:
    backend: dict[str, Any]
    search: dict[str, Any]
    generation: dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "RuntimeConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            backend=payload.get("backend", {}),
            search=payload.get("search", {}),
            generation=payload.get("generation", {}),
        )

    def with_overrides(
        self,
        backend_provider: str | None = None,
        model: str | None = None,
        search_provider: str | None = None,
    ) -> "RuntimeConfig":
        backend = dict(self.backend)
        search = dict(self.search)
        if backend_provider:
            backend["provider"] = backend_provider
        if model:
            backend["model"] = model
        if search_provider:
            search["provider"] = search_provider
        return RuntimeConfig(
            backend=backend,
            search=search,
            generation=dict(self.generation),
        )

    def temperature_for(self, stage: str) -> float:
        stage_temperatures = self.generation.get("stage_temperatures", {})
        if stage in stage_temperatures:
            return float(stage_temperatures[stage])
        return float(self.generation.get("default_temperature", 0.2))


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
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    source_id: int | None = None
    domain: str = ""
    published_at: str | None = None
    source_type: str = "web"
    provider: str = ""
    summary: str = ""
    text: str = ""
    highlights: list[str] = field(default_factory=list)


@dataclass
class SearchTrace:
    query: str
    mode: str
    provider: str
    results: list[SearchResult]
    error: str | None = None


@dataclass
class SearchRequest:
    query: str
    mode: str
    max_results: int
    category: str | None = None
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    user_location: str | None = None
    language: str | None = None


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
    search_traces: list[SearchTrace]
    events: list[StageEvent]
    metrics: dict[str, float]

