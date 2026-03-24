from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.critique.harper_validator import validate_harper_output
from src.critique.validator import validate_critique, validate_total_critique_budget
from src.orchestrator.backend import Backend
from src.orchestrator.types import (
    ProtocolConfig,
    RunRecord,
    RuntimeConfig,
    SearchRequest,
    SearchResult,
    SearchTrace,
    StageEvent,
)
from src.tools.json_utils import parse_json_from_text
from src.tools.token_utils import count_tokens_approx
from src.tools.web_search import SearchTool


class MultiAgentOrchestrator:
    def __init__(
        self,
        root: Path,
        config: ProtocolConfig,
        runtime_config: RuntimeConfig,
        backend: Backend,
        search_tool: SearchTool,
    ) -> None:
        self.root = root
        self.config = config
        self.runtime_config = runtime_config
        self.backend = backend
        self.search_tool = search_tool
        self.system_prompt = self._load_prompt("prompts/protocol/system.md")
        self.critique_schema_prompt = self._load_prompt("prompts/protocol/critique_schema.md")

    def _load_prompt(self, rel_path: str) -> str:
        return (self.root / rel_path).read_text(encoding="utf-8").strip()

    def _load_role_prompt(self, filename: str) -> str:
        return (self.root / "prompts" / "roles" / filename).read_text(encoding="utf-8").strip()

    def _max_tokens(self, key: str, fallback_key: str | None = None) -> int:
        if key in self.config.max_tokens:
            return int(self.config.max_tokens[key])
        if fallback_key and fallback_key in self.config.max_tokens:
            return int(self.config.max_tokens[fallback_key])
        raise KeyError(f"missing max token config for {key}")

    def _generate(
        self,
        role: str,
        stage: str,
        prompt: str,
        max_token_key: str,
        seed: int,
        fallback_key: str | None = None,
    ) -> str:
        return self.backend.generate(
            role=role,
            stage=stage,
            system_prompt=self.system_prompt,
            prompt=prompt,
            max_tokens=self._max_tokens(max_token_key, fallback_key=fallback_key),
            seed=seed,
            temperature=self.runtime_config.temperature_for(stage),
        )

    def _event(
        self,
        stage: str,
        role: str,
        start: float,
        text: str,
        retries: int = 0,
        accepted: bool | None = None,
        errors: list[str] | None = None,
    ) -> StageEvent:
        latency_ms = max(1, int((time.perf_counter() - start) * 1000))
        return StageEvent(
            stage=stage,
            role=role,
            latency_ms=latency_ms,
            token_count=count_tokens_approx(text),
            retries=retries,
            accepted=accepted,
            errors=errors,
        )

    def _default_captain_plan(self, user_query: str) -> dict[str, Any]:
        return {
            "overall_strategy": (
                "Answer the user query by combining targeted research, rigorous reasoning, "
                "and a review pass before synthesis."
            ),
            "harper_task": f"Search for evidence, implementation examples, and current context for: {user_query}",
            "benjamin_task": f"Stress-test the architecture and produce a rigorous plan for: {user_query}",
            "lucas_task": "Review Harper and Benjamin, flag gaps, and improve usability and tone.",
            "synthesis_checks": [
                "Separate facts from inference",
                "Call out external dependencies and failure modes",
                "Prefer an immediately runnable first iteration",
            ],
            "deliverable_shape": "practical answer with explicit next steps",
        }

    def _parse_captain_plan(self, raw: str, user_query: str) -> dict[str, Any]:
        try:
            payload = parse_json_from_text(raw)
        except ValueError:
            return self._default_captain_plan(user_query)

        if not isinstance(payload, dict):
            return self._default_captain_plan(user_query)

        plan = self._default_captain_plan(user_query)
        for key in plan:
            if key in payload:
                plan[key] = payload[key]
        if not isinstance(plan.get("synthesis_checks"), list):
            plan["synthesis_checks"] = self._default_captain_plan(user_query)["synthesis_checks"]
        return plan

    def _default_search_queries(self, user_query: str) -> list[str]:
        return [
            f"{user_query} implementation pattern",
            f"{user_query} qwen openrouter",
        ][: int(self.runtime_config.search.get("max_queries", 2))]

    def _default_search_plan(self, user_query: str) -> dict[str, Any]:
        return {
            "mode": str(self.runtime_config.search.get("default_mode", "fast_lookup")),
            "queries": self._default_search_queries(user_query),
            "category": None,
            "include_domains": [],
            "exclude_domains": [],
            "notes": "Use targeted web lookup before synthesis.",
        }

    def _parse_search_plan(self, raw: str, user_query: str) -> dict[str, Any]:
        try:
            payload = parse_json_from_text(raw)
        except ValueError:
            return self._default_search_plan(user_query)

        if isinstance(payload, list):
            plan = self._default_search_plan(user_query)
            plan["queries"] = [str(item).strip() for item in payload if str(item).strip()]
            return plan

        if not isinstance(payload, dict):
            return self._default_search_plan(user_query)

        plan = self._default_search_plan(user_query)
        queries = [str(item).strip() for item in payload.get("queries", []) if str(item).strip()]
        if not queries:
            queries = plan["queries"]
        limit = int(self.runtime_config.search.get("max_queries", 3))
        deduped: list[str] = []
        for query in queries:
            if query not in deduped:
                deduped.append(query)
        mode = str(payload.get("mode", plan["mode"]))
        if mode not in {"fast_lookup", "standard_research", "deep_research"}:
            mode = plan["mode"]
        include_domains = [
            str(item).strip().lower()
            for item in payload.get("include_domains", [])
            if str(item).strip()
        ]
        exclude_domains = [
            str(item).strip().lower()
            for item in payload.get("exclude_domains", [])
            if str(item).strip()
        ]
        return {
            "mode": mode,
            "queries": deduped[:limit],
            "category": payload.get("category"),
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "notes": str(payload.get("notes", plan["notes"])),
        }

    def _build_evidence_ledger(self, traces: list[SearchTrace]) -> list[SearchResult]:
        by_url: dict[str, SearchResult] = {}
        next_source_id = 1
        for trace in traces:
            for result in trace.results:
                if not result.url:
                    continue
                existing = by_url.get(result.url)
                if existing is None:
                    result.source_id = next_source_id
                    next_source_id += 1
                    by_url[result.url] = result
                    continue
                if not existing.snippet and result.snippet:
                    existing.snippet = result.snippet
                if not existing.summary and result.summary:
                    existing.summary = result.summary
                if not existing.text and result.text:
                    existing.text = result.text
                for highlight in result.highlights:
                    if highlight and highlight not in existing.highlights:
                        existing.highlights.append(highlight)
        return sorted(by_url.values(), key=lambda item: item.source_id or 0)

    def _serialize_search_traces(self, traces: list[SearchTrace]) -> str:
        if not traces:
            return "No search results available."
        rows: list[str] = []
        for trace in traces:
            rows.append(f"Query: {trace.query}")
            rows.append(f"Mode: {trace.mode}")
            rows.append(f"Provider: {trace.provider}")
            if trace.error:
                rows.append(f"Error: {trace.error}")
            if not trace.results:
                rows.append("Results: none")
                continue
            for idx, result in enumerate(trace.results, start=1):
                rows.append(f"[{idx}] {result.title}")
                rows.append(f"URL: {result.url}")
                rows.append(f"Snippet: {result.snippet}")
                if result.highlights:
                    rows.append(f"Highlights: {' | '.join(result.highlights[:2])}")
        return "\n".join(rows)

    def _serialize_evidence_ledger(self, ledger: list[SearchResult]) -> str:
        if not ledger:
            return "No evidence available."
        rows: list[str] = []
        for item in ledger:
            rows.append(f"[{item.source_id}] {item.title}")
            rows.append(f"URL: {item.url}")
            rows.append(f"Provider: {item.provider or item.source}")
            rows.append(f"Domain: {item.domain or 'unknown'}")
            if item.published_at:
                rows.append(f"Published: {item.published_at}")
            rows.append(f"Snippet: {item.snippet or 'none'}")
            if item.summary:
                rows.append(f"Summary: {item.summary}")
            if item.highlights:
                rows.append(f"Highlights: {' | '.join(item.highlights[:3])}")
        return "\n".join(rows)

    def _default_harper_output(self, user_query: str, ledger: list[SearchResult]) -> dict[str, Any]:
        digest = [
            {
                "source_id": item.source_id,
                "title": item.title,
                "url": item.url,
                "why_it_matters": (
                    item.summary
                    or item.snippet
                    or f"Potentially relevant evidence for {user_query}."
                )[:220],
            }
            for item in ledger[: min(3, len(ledger))]
        ]
        claim_sources = [item["source_id"] for item in digest if isinstance(item.get("source_id"), int)]
        if claim_sources:
            claims = [
                {
                    "claim": f"Available search evidence covers implementation context for: {user_query}",
                    "support": "partial",
                    "confidence": 0.55,
                    "sources": claim_sources[:2],
                    "notes": "Fallback summary because Harper JSON validation did not pass.",
                }
            ]
            open_questions = ["Need a stronger grounded synthesis from Harper once schema-valid output is available."]
        else:
            claims = [
                {
                    "claim": f"No usable web evidence was collected for: {user_query}",
                    "support": "uncertain",
                    "confidence": 0.2,
                    "sources": [],
                    "notes": "Search providers returned no usable results.",
                }
            ]
            open_questions = ["Need live search credentials or broader queries."]
        return {
            "answer_summary": "Harper fallback output based on the current evidence ledger.",
            "claims": claims,
            "source_digest": digest,
            "open_questions": open_questions[: int(self.config.harper.get("max_open_questions", 2))],
        }

    def run(self, run_id: str, task_id: str, user_query: str, seed: int) -> RunRecord:
        events: list[StageEvent] = []
        search_traces: list[SearchTrace] = []

        captain_plan_prompt = f"{self._load_role_prompt('captain_plan.md')}\n\nUser query:\n{user_query}"
        t0 = time.perf_counter()
        captain_plan_raw = self._generate(
            role="captain",
            stage="captain_plan",
            prompt=captain_plan_prompt,
            max_token_key="captain_plan",
            seed=seed,
        )
        events.append(self._event("captain_plan", "captain", t0, captain_plan_raw))
        captain_plan_structured = self._parse_captain_plan(captain_plan_raw, user_query)
        captain_plan = json.dumps(captain_plan_structured, indent=2)

        harper_search_plan_prompt = (
            f"{self._load_role_prompt('harper_search_plan.md')}\n\n"
            f"Captain plan:\n{captain_plan}\n\n"
            f"Assigned Harper task:\n{captain_plan_structured['harper_task']}\n\n"
            f"User query:\n{user_query}"
        )
        t_search_plan = time.perf_counter()
        raw_search_plan = self._generate(
            role="harper",
            stage="harper_search_plan",
            prompt=harper_search_plan_prompt,
            max_token_key="harper_search_plan",
            seed=seed,
        )
        events.append(self._event("harper_search_plan", "harper", t_search_plan, raw_search_plan))
        search_plan = self._parse_search_plan(raw_search_plan, user_query)

        max_results = int(self.runtime_config.search.get("max_results", 5))
        for query in search_plan["queries"]:
            t_search = time.perf_counter()
            try:
                trace = self.search_tool.search(
                    SearchRequest(
                        query=query,
                        mode=str(search_plan["mode"]),
                        max_results=max_results,
                        category=(
                            str(search_plan["category"])
                            if search_plan.get("category") is not None
                            else None
                        ),
                        include_domains=list(search_plan.get("include_domains", [])),
                        exclude_domains=list(search_plan.get("exclude_domains", [])),
                        user_location=str(self.runtime_config.search.get("user_location", "us")),
                        language=str(self.runtime_config.search.get("language", "en")),
                    )
                )
            except Exception as exc:
                trace = SearchTrace(
                    query=query,
                    mode=str(search_plan["mode"]),
                    provider="error",
                    results=[],
                    error=f"{type(exc).__name__}:{exc}",
                )
            search_traces.append(trace)
            text = json.dumps(asdict(trace), indent=2)
            events.append(
                self._event(
                    "harper_search",
                    "harper",
                    t_search,
                    text,
                    accepted=bool(trace.results),
                    errors=[trace.error] if trace.error else [],
                )
            )

        evidence_ledger = self._build_evidence_ledger(search_traces)

        role_outputs: dict[str, str] = {}
        harper_prompt = (
            f"{self._load_role_prompt('harper.md')}\n\n"
            f"Captain plan:\n{captain_plan}\n\n"
            f"Assigned Harper task:\n{captain_plan_structured['harper_task']}\n\n"
            f"Harper search plan:\n{json.dumps(search_plan, indent=2)}\n\n"
            f"Web search findings:\n{self._serialize_search_traces(search_traces)}\n\n"
            f"Evidence ledger:\n{self._serialize_evidence_ledger(evidence_ledger)}\n\n"
            f"User query:\n{user_query}"
        )
        valid_source_ids = {item.source_id for item in evidence_ledger if item.source_id is not None}
        harper_retries = 0
        harper_errors: list[str] = []
        harper_output_payload = self._default_harper_output(user_query, evidence_ledger)
        while harper_retries <= self.config.retry["max_schema_retries"]:
            t_harper = time.perf_counter()
            harper_raw = self._generate(
                role="harper",
                stage="harper_research",
                prompt=harper_prompt,
                max_token_key="harper_output",
                fallback_key="role_output",
                seed=seed + harper_retries,
            )
            validation = validate_harper_output(
                raw=harper_raw,
                valid_source_ids={int(item) for item in valid_source_ids},
                max_claims=int(self.config.harper.get("max_claims", 6)),
                max_open_questions=int(self.config.harper.get("max_open_questions", 2)),
                max_tokens=int(self.config.max_tokens["harper_output"]),
            )
            events.append(
                self._event(
                    "harper_research",
                    "harper",
                    t_harper,
                    harper_raw,
                    retries=harper_retries,
                    accepted=validation.accepted,
                    errors=validation.errors,
                )
            )
            if validation.accepted:
                harper_output_payload = validation.payload
                break
            harper_errors = validation.errors
            harper_retries += 1

        if harper_retries > self.config.retry["max_schema_retries"] and harper_errors:
            fallback_text = json.dumps(harper_output_payload, indent=2)
            events.append(
                StageEvent(
                    stage="harper_fallback",
                    role="harper",
                    latency_ms=0,
                    token_count=count_tokens_approx(fallback_text),
                    retries=harper_retries,
                    accepted=False,
                    errors=harper_errors,
                )
            )

        harper_output = json.dumps(harper_output_payload, indent=2)
        role_outputs["harper"] = harper_output

        benjamin_prompt = (
            f"{self._load_role_prompt('benjamin.md')}\n\n"
            f"Captain plan:\n{captain_plan}\n\n"
            f"Assigned Benjamin task:\n{captain_plan_structured['benjamin_task']}\n\n"
            f"User query:\n{user_query}"
        )
        t_benjamin = time.perf_counter()
        benjamin_output = self._generate(
            role="benjamin",
            stage="benjamin_reasoning",
            prompt=benjamin_prompt,
            max_token_key="benjamin_output",
            fallback_key="role_output",
            seed=seed,
        )
        role_outputs["benjamin"] = benjamin_output
        events.append(self._event("benjamin_reasoning", "benjamin", t_benjamin, benjamin_output))

        lucas_prompt = (
            f"{self._load_role_prompt('lucas.md')}\n\n"
            f"Captain plan:\n{captain_plan}\n\n"
            f"Assigned Lucas task:\n{captain_plan_structured['lucas_task']}\n\n"
            f"Harper findings:\n{harper_output}\n\n"
            f"Benjamin findings:\n{benjamin_output}\n\n"
            f"User query:\n{user_query}"
        )
        t_lucas = time.perf_counter()
        lucas_output = self._generate(
            role="lucas",
            stage="lucas_review",
            prompt=lucas_prompt,
            max_token_key="lucas_output",
            fallback_key="role_output",
            seed=seed,
        )
        role_outputs["lucas"] = lucas_output
        events.append(self._event("lucas_review", "lucas", t_lucas, lucas_output))

        critiques: dict[str, dict[str, str]] = {}
        accepted_count = 0
        allowed_types = set(self.config.critique["allowed_challenge_types"])
        retries = 0
        errors: list[str] = []
        accepted = False
        critique_json: dict[str, str] = {}
        while retries <= self.config.retry["max_schema_retries"]:
            critique_prompt = (
                f"{self._load_role_prompt('critique.md').format(role='lucas')}\n\n"
                f"{self.critique_schema_prompt}\n\n"
                f"Captain plan:\n{captain_plan}\n\n"
                f"Role outputs:\n{json.dumps(role_outputs, indent=2)}\n\n"
                f"User query:\n{user_query}"
            )
            t_critique = time.perf_counter()
            raw = self._generate(
                role="lucas",
                stage="lucas_critique",
                prompt=critique_prompt,
                max_token_key="critique",
                seed=seed + retries,
            )
            result = validate_critique(
                raw=raw,
                allowed_types=allowed_types,
                target_max_tokens=int(self.config.critique["target_max_tokens"]),
                correction_max_tokens=int(self.config.critique["correction_max_tokens"]),
                critique_max_tokens=int(self.config.max_tokens["critique"]),
            )
            events.append(
                self._event(
                    stage="lucas_critique",
                    role="lucas",
                    start=t_critique,
                    text=raw,
                    retries=retries,
                    accepted=result.accepted,
                    errors=result.errors,
                )
            )
            if result.accepted:
                critique_json = result.critique
                accepted = True
                break
            errors = result.errors
            retries += 1

        if not accepted:
            critique_json = {
                "challenge_type": "missing_consideration",
                "target": "lucas_critique_generation_failure",
                "correction": "use conservative captain synthesis with explicit uncertainty",
            }
            events.append(
                StageEvent(
                    stage="critique_fallback",
                    role="lucas",
                    latency_ms=0,
                    token_count=count_tokens_approx(json.dumps(critique_json)),
                    retries=retries,
                    accepted=False,
                    errors=errors,
                )
            )
        else:
            accepted_count += 1
        critiques["lucas"] = critique_json

        critique_values = list(critiques.values())
        if not validate_total_critique_budget(
            critique_values, self.config.max_tokens["total_critique_budget"]
        ):
            for role, critique in critiques.items():
                correction_words = critique["correction"].split()
                critiques[role]["correction"] = " ".join(correction_words[:15])

        synthesis_prompt = (
            f"{self._load_role_prompt('captain_synthesis.md')}\n\n"
            f"Captain plan:\n{captain_plan}\n\n"
            f"Harper search plan:\n{json.dumps(search_plan, indent=2)}\n\n"
            f"Search evidence:\n{self._serialize_search_traces(search_traces)}\n\n"
            f"Evidence ledger:\n{self._serialize_evidence_ledger(evidence_ledger)}\n\n"
            f"Role outputs:\n{json.dumps(role_outputs, indent=2)}\n\n"
            f"Accepted critiques:\n{json.dumps(critiques, indent=2)}\n\n"
            f"User query:\n{user_query}"
        )
        t_final = time.perf_counter()
        final_answer = self._generate(
            role="captain",
            stage="captain_synthesis",
            prompt=synthesis_prompt,
            max_token_key="captain_synthesis",
            seed=seed,
        )
        events.append(self._event("captain_synthesis", "captain", t_final, final_answer))

        total_tokens = sum(event.token_count for event in events)
        total_latency_ms = sum(event.latency_ms for event in events)
        metrics = {
            "total_tokens": float(total_tokens),
            "total_latency_ms": float(total_latency_ms),
            "critique_acceptance_rate": float(accepted_count / max(1, len(critiques))),
        }

        return RunRecord(
            run_id=run_id,
            task_id=task_id,
            seed=seed,
            user_query=user_query,
            captain_plan=captain_plan,
            role_outputs=role_outputs,
            critiques=critiques,
            final_answer=final_answer,
            search_traces=search_traces,
            events=events,
            metrics=metrics,
        )
