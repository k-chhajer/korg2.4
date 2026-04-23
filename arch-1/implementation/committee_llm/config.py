from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROLE_SEQUENCE = (
    "coordinator_plan",
    "researcher",
    "analyst",
    "critic",
    "coordinator_finalize",
)

BASE_ROLE_ORDER = (
    "coordinator",
    "researcher",
    "analyst",
    "critic",
    "coordinator",
)

DEFAULT_OUTPUT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "coordinator_plan": (
        "Problem Restatement",
        "Success Criteria",
        "Plan",
        "Risks and Unknowns",
        "Research Brief",
    ),
    "researcher": (
        "Evidence and Observations",
        "Options",
        "Assumptions",
        "Open Questions",
        "Research Handoff",
    ),
    "analyst": (
        "Core Answer",
        "Reasoning",
        "Caveats",
        "Draft Deliverable",
    ),
    "critic": (
        "Major Issues",
        "Minor Issues",
        "Missing Evidence",
        "Required Revisions",
    ),
    "coordinator_finalize": (
        "Final Answer",
        "Decision Rationale",
        "Residual Risks",
    ),
}

DEFAULT_LOCAL_CONTEXT_VIEWS: dict[str, tuple[str, ...]] = {
    "coordinator_plan": (
        "task_prompt",
        "task_domain",
        "benchmark_name",
        "benchmark_split",
        "task_type",
        "task_context",
        "task_objectives",
        "task_constraints",
        "answer_choices",
        "deliverable",
        "evaluation_criteria",
        "provided_references",
    ),
    "researcher": (
        "task_prompt",
        "task_domain",
        "benchmark_name",
        "benchmark_split",
        "task_type",
        "task_context",
        "task_objectives",
        "task_constraints",
        "answer_choices",
        "deliverable",
        "evaluation_criteria",
        "provided_references",
        "coordinator_plan",
    ),
    "analyst": (
        "task_prompt",
        "task_domain",
        "benchmark_name",
        "benchmark_split",
        "task_type",
        "task_context",
        "task_objectives",
        "task_constraints",
        "answer_choices",
        "deliverable",
        "evaluation_criteria",
        "provided_references",
        "coordinator_plan",
        "researcher_notes",
    ),
    "critic": (
        "task_prompt",
        "task_domain",
        "benchmark_name",
        "benchmark_split",
        "task_type",
        "task_context",
        "task_objectives",
        "task_constraints",
        "answer_choices",
        "deliverable",
        "evaluation_criteria",
        "provided_references",
        "coordinator_plan",
        "researcher_notes",
        "analyst_draft",
    ),
    "coordinator_finalize": (
        "task_prompt",
        "task_domain",
        "benchmark_name",
        "benchmark_split",
        "task_type",
        "task_context",
        "task_objectives",
        "task_constraints",
        "answer_choices",
        "deliverable",
        "evaluation_criteria",
        "provided_references",
        "coordinator_plan",
        "researcher_notes",
        "analyst_draft",
        "critic_feedback",
    ),
}


@dataclass(slots=True)
class ProviderConfig:
    api_base: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: int = 180


@dataclass(slots=True)
class DefaultRoleSettings:
    model: str
    temperature: float = 0.3
    max_tokens: int = 1200
    extra_body: dict[str, Any] | None = None


@dataclass(slots=True)
class RoleConfig:
    system_prompt_path: Path
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] | None = None
    output_schema: tuple[str, ...] = ()
    local_context_view: tuple[str, ...] = ()
    role_specialization: str | None = None
    checkpoint_name: str | None = None


@dataclass(slots=True)
class ArchitectureSpec:
    family: str
    description: str
    specialization_source: str
    stage_sequence: tuple[str, ...]
    max_revision_loops: int = 1
    allow_early_stop: bool = False
    paper_reference_role_order: tuple[str, ...] = BASE_ROLE_ORDER


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    provider: ProviderConfig
    defaults: DefaultRoleSettings
    roles: dict[str, RoleConfig]
    architecture: ArchitectureSpec
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source_path = Path(path).resolve()
        raw = json.loads(source_path.read_text(encoding="utf-8"))

        provider = ProviderConfig(**raw["provider"])
        defaults = DefaultRoleSettings(**raw["defaults"])

        architecture_raw = dict(raw.get("architecture") or {})
        stage_sequence = tuple(architecture_raw.get("stage_sequence") or ROLE_SEQUENCE)
        if stage_sequence != ROLE_SEQUENCE:
            raise ValueError(
                "Architecture 1 requires the fixed stage sequence "
                "'coordinator_plan -> researcher -> analyst -> critic -> coordinator_finalize'."
            )

        architecture = ArchitectureSpec(
            family=str(architecture_raw.get("family") or "architecture_1"),
            description=str(
                architecture_raw.get("description")
                or "Role-specialized committee with a fixed coordinator->researcher->analyst->critic->coordinator flow."
            ),
            specialization_source=str(
                architecture_raw.get("specialization_source") or "prompt_scaffold_only"
            ),
            stage_sequence=stage_sequence,
            max_revision_loops=int(architecture_raw.get("max_revision_loops", 1)),
            allow_early_stop=bool(architecture_raw.get("allow_early_stop", False)),
            paper_reference_role_order=tuple(
                architecture_raw.get("paper_reference_role_order") or BASE_ROLE_ORDER
            ),
        )

        roles: dict[str, RoleConfig] = {}
        for role_name in ROLE_SEQUENCE:
            if role_name not in raw["roles"]:
                raise ValueError(f"Missing required role '{role_name}' in {source_path}")

            role_raw = dict(raw["roles"][role_name])
            prompt_path = Path(role_raw.pop("system_prompt_path"))
            if not prompt_path.is_absolute():
                prompt_path = (source_path.parent / prompt_path).resolve()

            output_schema = tuple(role_raw.pop("output_schema", DEFAULT_OUTPUT_SCHEMAS[role_name]))
            local_context_view = tuple(role_raw.pop("local_context_view", DEFAULT_LOCAL_CONTEXT_VIEWS[role_name]))

            roles[role_name] = RoleConfig(
                system_prompt_path=prompt_path,
                output_schema=output_schema,
                local_context_view=local_context_view,
                **role_raw,
            )

        return cls(
            name=raw["name"],
            provider=provider,
            defaults=defaults,
            roles=roles,
            architecture=architecture,
            source_path=source_path,
        )

    def resolve_role_settings(self, role_name: str) -> dict[str, str | float | int | dict[str, Any] | None]:
        role = self.roles[role_name]
        return {
            "model": role.model or self.defaults.model,
            "temperature": role.temperature
            if role.temperature is not None
            else self.defaults.temperature,
            "max_tokens": role.max_tokens
            if role.max_tokens is not None
            else self.defaults.max_tokens,
            "extra_body": role.extra_body if role.extra_body is not None else self.defaults.extra_body,
        }

    def load_system_prompt(self, role_name: str) -> str:
        prompt_path = self.roles[role_name].system_prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8").strip()

    def get_role_schema(self, role_name: str) -> tuple[str, ...]:
        schema = self.roles[role_name].output_schema
        if not schema:
            raise ValueError(f"Role '{role_name}' must declare a non-empty output schema.")
        return schema

    def get_local_context_view(self, role_name: str) -> tuple[str, ...]:
        context_view = self.roles[role_name].local_context_view
        if not context_view:
            raise ValueError(f"Role '{role_name}' must declare a non-empty local context view.")
        return context_view

    def paper_claim_blockers(self) -> list[str]:
        blockers: list[str] = []

        if self.architecture.family != "architecture_1":
            blockers.append("Config is not marked as Architecture 1.")

        if self.architecture.paper_reference_role_order != BASE_ROLE_ORDER:
            blockers.append("Paper reference role order does not match coordinator->researcher->analyst->critic->coordinator.")

        if self.architecture.stage_sequence != ROLE_SEQUENCE:
            blockers.append("Stage sequence does not match the fixed Architecture 1 topology.")

        if self.architecture.specialization_source != "post_trained_role_specialists":
            blockers.append(
                "Role specialization source is not set to post_trained_role_specialists, so this config should be treated as a prompt scaffold."
            )

        seen_checkpoints: set[str] = set()
        for role_name in ROLE_SEQUENCE:
            role = self.roles[role_name]
            if not role.role_specialization:
                blockers.append(f"Role '{role_name}' is missing role_specialization metadata.")
            if not role.output_schema:
                blockers.append(f"Role '{role_name}' is missing an output schema.")
            if not role.local_context_view:
                blockers.append(f"Role '{role_name}' is missing a local context view.")
            if self.architecture.specialization_source == "post_trained_role_specialists":
                if not role.checkpoint_name:
                    blockers.append(f"Role '{role_name}' is missing a role-specific checkpoint_name.")
                elif role.checkpoint_name in seen_checkpoints:
                    blockers.append(
                        f"Checkpoint '{role.checkpoint_name}' is reused across roles; Architecture 1 requires role-specific post-training."
                    )
                else:
                    seen_checkpoints.add(role.checkpoint_name)

        return blockers
