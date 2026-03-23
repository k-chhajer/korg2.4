from __future__ import annotations

import json
import time
from pathlib import Path

from src.critique.validator import validate_critique, validate_total_critique_budget
from src.orchestrator.backend import Backend
from src.orchestrator.types import ProtocolConfig, RunRecord, StageEvent
from src.tools.token_utils import count_tokens_approx


class MultiAgentOrchestrator:
    def __init__(self, root: Path, config: ProtocolConfig, backend: Backend) -> None:
        self.root = root
        self.config = config
        self.backend = backend
        self.system_prompt = self._load_prompt("prompts/protocol/system.md")
        self.critique_schema_prompt = self._load_prompt("prompts/protocol/critique_schema.md")

    def _load_prompt(self, rel_path: str) -> str:
        return (self.root / rel_path).read_text(encoding="utf-8").strip()

    def _load_role_prompt(self, filename: str) -> str:
        return (self.root / "prompts" / "roles" / filename).read_text(encoding="utf-8").strip()

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

    def run(self, run_id: str, task_id: str, user_query: str, seed: int) -> RunRecord:
        events: list[StageEvent] = []

        captain_plan_prompt = (
            f"{self.system_prompt}\n\n"
            f"{self._load_role_prompt('captain_plan.md')}\n\n"
            f"User query:\n{user_query}"
        )
        t0 = time.perf_counter()
        captain_plan = self.backend.generate(
            role="captain",
            stage="captain_plan",
            prompt=captain_plan_prompt,
            max_tokens=self.config.max_tokens["captain_plan"],
            seed=seed,
        )
        events.append(self._event("captain_plan", "captain", t0, captain_plan))

        role_outputs: dict[str, str] = {}
        for role in ("harper", "benjamin", "lucas"):
            role_prompt = (
                f"{self.system_prompt}\n\n"
                f"{self._load_role_prompt(f'{role}.md')}\n\n"
                f"Captain plan:\n{captain_plan}\n\n"
                f"User query:\n{user_query}"
            )
            t_role = time.perf_counter()
            text = self.backend.generate(
                role=role,
                stage="role_output",
                prompt=role_prompt,
                max_tokens=self.config.max_tokens["role_output"],
                seed=seed,
            )
            role_outputs[role] = text
            events.append(self._event("role_output", role, t_role, text))

        critiques: dict[str, dict[str, str]] = {}
        accepted_count = 0
        allowed_types = set(self.config.critique["allowed_challenge_types"])
        for role in ("harper", "benjamin", "lucas"):
            retries = 0
            errors: list[str] = []
            accepted = False
            critique_json: dict[str, str] = {}
            while retries <= self.config.retry["max_schema_retries"]:
                critique_prompt = (
                    f"{self.system_prompt}\n\n"
                    f"{self._load_role_prompt('critique.md').format(role=role)}\n\n"
                    f"{self.critique_schema_prompt}\n\n"
                    f"Captain plan:\n{captain_plan}\n\n"
                    f"Role outputs:\n{json.dumps(role_outputs, indent=2)}\n\n"
                    f"User query:\n{user_query}"
                )
                t_critique = time.perf_counter()
                raw = self.backend.generate(
                    role=role,
                    stage="critique",
                    prompt=critique_prompt,
                    max_tokens=self.config.max_tokens["critique"],
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
                        stage="critique",
                        role=role,
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
                    "target": "critique_generation_failure",
                    "correction": "use conservative captain synthesis with explicit uncertainty",
                }
                events.append(
                    StageEvent(
                        stage="critique_fallback",
                        role=role,
                        latency_ms=0,
                        token_count=count_tokens_approx(json.dumps(critique_json)),
                        retries=retries,
                        accepted=False,
                        errors=errors,
                    )
                )
            else:
                accepted_count += 1
            critiques[role] = critique_json

        critique_values = list(critiques.values())
        if not validate_total_critique_budget(
            critique_values, self.config.max_tokens["total_critique_budget"]
        ):
            # Shrink corrections to enforce a hard budget.
            for role, critique in critiques.items():
                correction_words = critique["correction"].split()
                critiques[role]["correction"] = " ".join(correction_words[:15])

        synthesis_prompt = (
            f"{self.system_prompt}\n\n"
            f"{self._load_role_prompt('captain_synthesis.md')}\n\n"
            f"Captain plan:\n{captain_plan}\n\n"
            f"Role outputs:\n{json.dumps(role_outputs, indent=2)}\n\n"
            f"Accepted critiques:\n{json.dumps(critiques, indent=2)}\n\n"
            f"User query:\n{user_query}"
        )
        t_final = time.perf_counter()
        final_answer = self.backend.generate(
            role="captain",
            stage="captain_synthesis",
            prompt=synthesis_prompt,
            max_tokens=self.config.max_tokens["captain_synthesis"],
            seed=seed,
        )
        events.append(self._event("captain_synthesis", "captain", t_final, final_answer))

        total_tokens = sum(event.token_count for event in events)
        total_latency_ms = sum(event.latency_ms for event in events)
        metrics = {
            "total_tokens": float(total_tokens),
            "total_latency_ms": float(total_latency_ms),
            "critique_acceptance_rate": float(accepted_count / 3.0),
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
            events=events,
            metrics=metrics,
        )
