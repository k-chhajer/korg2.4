from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig
from committee_llm.evaluation import evaluate_answer
from committee_llm.metrics import aggregate_usage
from committee_llm.tasks import TaskSpec


STAGE_CONTEXT_ALIASES = {
    "researcher_notes": "researcher",
    "analyst_draft": "analyst",
    "critic_feedback": "critic",
}

KNOWN_HEADERS = sorted(
    {
        "Problem Restatement",
        "Success Criteria",
        "Plan",
        "Risks and Unknowns",
        "Research Brief",
        "Evidence and Observations",
        "Options",
        "Assumptions",
        "Open Questions",
        "Research Handoff",
        "Core Answer",
        "Reasoning",
        "Caveats",
        "Draft Deliverable",
        "Major Issues",
        "Minor Issues",
        "Missing Evidence",
        "Required Revisions",
        "Final Answer",
        "Decision Rationale",
        "Residual Risks",
    },
    key=len,
    reverse=True,
)


def _tag_block(name: str, content: str) -> str:
    return f"<{name}>\n{content.strip()}\n</{name}>"


def _tag_list_block(name: str, items: list[str]) -> str:
    return _tag_block(name, "\n".join(f"- {item}" for item in items))


def _extract_section(text: str, header: str) -> str:
    marker = f"{header}:"
    if marker not in text:
        return ""

    _, tail = text.split(marker, 1)
    end_index = len(tail)
    for next_header in KNOWN_HEADERS:
        if next_header == header:
            continue
        candidate = tail.find(f"\n{next_header}:")
        if candidate != -1 and candidate < end_index:
            end_index = candidate

    return tail[:end_index].strip()


def _extract_schema_sections(text: str, schema: tuple[str, ...]) -> dict[str, str]:
    return {header: _extract_section(text, header) for header in schema}


def _schema_validity(text: str, schema: tuple[str, ...]) -> dict[str, Any]:
    sections = _extract_schema_sections(text, schema)
    missing = [header for header, value in sections.items() if not value]
    return {
        "required_headers": list(schema),
        "present_headers": [header for header, value in sections.items() if value],
        "missing_headers": missing,
        "is_valid": not missing,
        "sections": sections,
    }


def _build_task_blocks(task: TaskSpec) -> dict[str, str]:
    blocks = {"task_prompt": _tag_block("task_prompt", task.prompt)}

    if task.domain:
        blocks["task_domain"] = _tag_block("task_domain", task.domain)
    if task.benchmark_name:
        blocks["benchmark_name"] = _tag_block("benchmark_name", task.benchmark_name)
    if task.benchmark_split:
        blocks["benchmark_split"] = _tag_block("benchmark_split", task.benchmark_split)
    if task.task_type:
        blocks["task_type"] = _tag_block("task_type", task.task_type)
    if task.context:
        blocks["task_context"] = _tag_block("task_context", task.context)
    if task.objectives:
        blocks["task_objectives"] = _tag_list_block("task_objectives", task.objectives)
    if task.constraints:
        blocks["task_constraints"] = _tag_list_block("task_constraints", task.constraints)
    if task.choices:
        blocks["answer_choices"] = _tag_list_block(
            "answer_choices",
            [f"{choice['label']}. {choice['text']}" for choice in task.choices],
        )
    if task.deliverable:
        blocks["deliverable"] = _tag_block("deliverable", task.deliverable)
    if task.evaluation_criteria:
        blocks["evaluation_criteria"] = _tag_list_block("evaluation_criteria", task.evaluation_criteria)
    if task.references:
        blocks["provided_references"] = _tag_list_block("provided_references", task.references)

    return blocks


def _build_stage_context_blocks(
    role_name: str,
    task: TaskSpec,
    stage_outputs: dict[str, str],
    config: ExperimentConfig,
) -> list[str]:
    task_blocks = _build_task_blocks(task)
    context_view = config.get_local_context_view(role_name)
    blocks: list[str] = []

    for block_name in context_view:
        if block_name in task_blocks:
            blocks.append(task_blocks[block_name])
            continue

        stage_name = STAGE_CONTEXT_ALIASES.get(block_name, block_name)
        if stage_name in stage_outputs:
            blocks.append(_tag_block(block_name, stage_outputs[stage_name]))

    return blocks


def _build_role_prompt(
    role_name: str,
    task: TaskSpec,
    stage_outputs: dict[str, str],
    config: ExperimentConfig,
) -> str:
    schema = config.get_role_schema(role_name)
    instructions = {
        "coordinator_plan": (
            "You are stage 1 of a fixed committee pipeline.",
            "Decompose the task for downstream specialists. Do not solve it yet.",
        ),
        "researcher": (
            "You are stage 2 of a fixed committee pipeline.",
            "Gather candidate evidence, relevant distinctions, and unresolved questions.",
        ),
        "analyst": (
            "You are stage 3 of a fixed committee pipeline.",
            "Convert the gathered material into a structured reasoning path and tentative solution.",
        ),
        "critic": (
            "You are stage 4 of a fixed committee pipeline.",
            "Stress-test the candidate solution, flag unsupported jumps, and request concrete revisions.",
        ),
        "coordinator_finalize": (
            "You are stage 5 of a fixed committee pipeline.",
            "Decide whether to accept the draft, integrate the valid critique, and produce the final answer.",
        ),
    }[role_name]

    return "\n\n".join(
        _build_stage_context_blocks(role_name, task, stage_outputs, config)
        + [
            instructions[0],
            instructions[1],
            (
                "Return exactly these section headers in order: "
                + ", ".join(f"{header}:" for header in schema)
                + "."
            ),
        ]
    )


def _classify_yes_no(text: str) -> bool | None:
    lowered = text.lower()
    if re.search(r"\b(yes|present|valid|grounded|required)\b", lowered):
        return True
    if re.search(r"\b(no|none|missing|invalid|unsupported|not needed)\b", lowered):
        return False
    return None


def _behavior_metrics(
    task: TaskSpec,
    stage_outputs: dict[str, str],
    stage_schemas: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    plan_sections = stage_schemas.get("coordinator_plan", {}).get("sections", {})
    research_sections = stage_schemas.get("researcher", {}).get("sections", {})
    analyst_sections = stage_schemas.get("analyst", {}).get("sections", {})
    critic_sections = stage_schemas.get("critic", {}).get("sections", {})
    final_sections = stage_schemas.get("coordinator_finalize", {}).get("sections", {})

    decomposition_rate = 1.0 if plan_sections.get("Plan") and len(plan_sections.get("Plan", "").splitlines()) >= 2 else 0.0
    evidence_grounding_rate = 1.0 if research_sections.get("Evidence and Observations") else 0.0
    revision_after_critique_rate = 1.0 if critic_sections.get("Required Revisions") and final_sections.get("Decision Rationale") else 0.0
    premature_finalization_rate = 0.0 if analyst_sections.get("Draft Deliverable") and final_sections.get("Final Answer") else 1.0

    final_text = stage_outputs.get("coordinator_finalize", "")
    unsupported_claim_rate = 0.0
    if task.context and final_text:
        normalized_context = task.context.lower()
        unsupported_claim_rate = 1.0 if not any(token in normalized_context for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{4,}", final_text.lower())[:12]) else 0.0

    return {
        "decomposition_rate": decomposition_rate,
        "evidence_grounding_rate": evidence_grounding_rate,
        "revision_after_critique_rate": revision_after_critique_rate,
        "premature_finalization_rate": premature_finalization_rate,
        "unsupported_claim_rate": unsupported_claim_rate,
    }


def _build_trace_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    per_role_elapsed_sec: dict[str, float] = {}
    per_role_usage: dict[str, dict[str, Any]] = {}

    for step in steps:
        role_name = str(step["role"])
        per_role_elapsed_sec[role_name] = float(step["elapsed_sec"])
        usage = step.get("usage")
        per_role_usage[role_name] = usage if isinstance(usage, dict) else {}

    return {
        "stage_count": len(steps),
        "model_call_count": len(steps),
        "per_role_elapsed_sec": per_role_elapsed_sec,
        "per_role_usage": per_role_usage,
        "usage": aggregate_usage(per_role_usage.values()),
    }


@dataclass(slots=True)
class CommitteeRunner:
    config: ExperimentConfig
    client: OpenAICompatibleChatClient

    def run(self, user_task: str, task_id: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        task = TaskSpec.from_prompt(user_task, task_id=task_id, metadata=metadata)
        return self.run_task(task)

    def run_task(self, task: TaskSpec) -> dict[str, Any]:
        run_id = task.task_id or str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        run_start = time.perf_counter()
        stage_outputs: dict[str, str] = {}
        stage_schemas: dict[str, dict[str, Any]] = {}
        steps: list[dict[str, Any]] = []

        for role_name in self.config.architecture.stage_sequence:
            role_settings = self.config.resolve_role_settings(role_name)
            role_config = self.config.roles[role_name]
            system_prompt = self.config.load_system_prompt(role_name)
            user_prompt = _build_role_prompt(role_name, task, stage_outputs, self.config)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            result = self.client.chat(
                model=str(role_settings["model"]),
                messages=messages,
                temperature=float(role_settings["temperature"]),
                max_tokens=int(role_settings["max_tokens"]),
            )

            schema_validation = _schema_validity(result.content, self.config.get_role_schema(role_name))
            stage_outputs[role_name] = result.content
            stage_schemas[role_name] = schema_validation
            steps.append(
                {
                    "role": role_name,
                    "role_specialization": role_config.role_specialization,
                    "checkpoint_name": role_config.checkpoint_name,
                    "model": result.model,
                    "temperature": role_settings["temperature"],
                    "max_tokens": role_settings["max_tokens"],
                    "system_prompt": system_prompt,
                    "local_context_view": list(self.config.get_local_context_view(role_name)),
                    "output_schema": list(self.config.get_role_schema(role_name)),
                    "schema_validation": schema_validation,
                    "user_prompt": user_prompt,
                    "response": result.content,
                    "elapsed_sec": result.elapsed_sec,
                    "usage": result.usage,
                }
            )

        final_stage = stage_outputs["coordinator_finalize"]
        final_answer = _extract_section(final_stage, "Final Answer") or final_stage.strip()
        summary = _build_trace_summary(steps)
        evaluation = evaluate_answer(task, final_answer)
        summary["behavior_metrics"] = _behavior_metrics(task, stage_outputs, stage_schemas)
        summary["schema_validity"] = {
            role_name: {
                "is_valid": payload["is_valid"],
                "missing_headers": payload["missing_headers"],
            }
            for role_name, payload in stage_schemas.items()
        }

        return {
            "run_id": run_id,
            "architecture": "architecture_1_posttrained_committee",
            "architecture_spec": {
                "family": self.config.architecture.family,
                "description": self.config.architecture.description,
                "specialization_source": self.config.architecture.specialization_source,
                "stage_sequence": list(self.config.architecture.stage_sequence),
                "paper_reference_role_order": list(self.config.architecture.paper_reference_role_order),
                "paper_claim_blockers": self.config.paper_claim_blockers(),
            },
            "config_name": self.config.name,
            "config_path": str(self.config.source_path),
            "started_at_utc": started_at,
            "total_elapsed_sec": time.perf_counter() - run_start,
            "input": task.prompt,
            "task": task.to_trace_payload(),
            "metadata": dict(task.metadata),
            "steps": steps,
            "final_stage": final_stage,
            "final_answer": final_answer,
            "evaluation": evaluation,
            "summary": summary,
        }
