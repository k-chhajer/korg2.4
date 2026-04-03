from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig, ROLE_SEQUENCE
from committee_llm.evaluation import evaluate_answer
from committee_llm.metrics import aggregate_usage
from committee_llm.tasks import TaskSpec


def _tag_block(name: str, content: str) -> str:
    return f"<{name}>\n{content.strip()}\n</{name}>"


def _tag_list_block(name: str, items: list[str]) -> str:
    return _tag_block(name, "\n".join(f"- {item}" for item in items))


def _extract_section(text: str, header: str) -> str:
    marker = f"{header}:"
    if marker not in text:
        return text.strip()

    _, tail = text.split(marker, 1)
    known_headers = [
        "Problem Restatement:",
        "Success Criteria:",
        "Plan:",
        "Risks and Unknowns:",
        "Research Brief:",
        "Evidence and Observations:",
        "Options:",
        "Assumptions:",
        "Open Questions:",
        "Research Handoff:",
        "Core Answer:",
        "Reasoning:",
        "Caveats:",
        "Draft Deliverable:",
        "Major Issues:",
        "Minor Issues:",
        "Missing Evidence:",
        "Required Revisions:",
        "Final Answer:",
        "Decision Rationale:",
        "Residual Risks:",
    ]

    end_index = len(tail)
    for next_header in known_headers:
        if next_header == marker:
            continue
        candidate = tail.find(f"\n{next_header}")
        if candidate != -1 and candidate < end_index:
            end_index = candidate

    return tail[:end_index].strip()


def _build_task_blocks(task: TaskSpec) -> list[str]:
    blocks = [_tag_block("task_prompt", task.prompt)]

    if task.domain:
        blocks.append(_tag_block("task_domain", task.domain))
    if task.benchmark_name:
        blocks.append(_tag_block("benchmark_name", task.benchmark_name))
    if task.benchmark_split:
        blocks.append(_tag_block("benchmark_split", task.benchmark_split))
    if task.task_type:
        blocks.append(_tag_block("task_type", task.task_type))
    if task.context:
        blocks.append(_tag_block("task_context", task.context))
    if task.objectives:
        blocks.append(_tag_list_block("task_objectives", task.objectives))
    if task.constraints:
        blocks.append(_tag_list_block("task_constraints", task.constraints))
    if task.choices:
        blocks.append(_tag_list_block("answer_choices", [f"{choice['label']}. {choice['text']}" for choice in task.choices]))
    if task.deliverable:
        blocks.append(_tag_block("deliverable", task.deliverable))
    if task.evaluation_criteria:
        blocks.append(_tag_list_block("evaluation_criteria", task.evaluation_criteria))
    if task.references:
        blocks.append(_tag_list_block("provided_references", task.references))

    return blocks


def _build_role_prompt(role_name: str, task: TaskSpec, stage_outputs: dict[str, str]) -> str:
    task_blocks = _build_task_blocks(task)

    if role_name == "coordinator_plan":
        return "\n\n".join(
            task_blocks
            + [
                "You are stage 1 of a fixed committee pipeline.",
                (
                    "Produce a decomposition that downstream agents can follow. "
                    "Treat the deliverable, constraints, and evaluation criteria as binding. "
                    "Return the exact section headers requested in your system prompt."
                ),
            ]
        )

    if role_name == "researcher":
        return "\n\n".join(
            task_blocks
            + [
                "You are stage 2 of a fixed committee pipeline.",
                _tag_block("coordinator_plan", stage_outputs["coordinator_plan"]),
                (
                    "Expand the plan into evidence, options, assumptions, and unresolved issues. "
                    "Prioritize information that helps satisfy the deliverable and evaluation criteria. "
                    "Do not write the final answer."
                ),
            ]
        )

    if role_name == "analyst":
        return "\n\n".join(
            task_blocks
            + [
                "You are stage 3 of a fixed committee pipeline.",
                _tag_block("coordinator_plan", stage_outputs["coordinator_plan"]),
                _tag_block("researcher_notes", stage_outputs["researcher"]),
                (
                    "Synthesize the research into the strongest draft answer you can. "
                    "Be decisive, but preserve caveats where evidence is weak. "
                    "Make the draft match the requested deliverable."
                ),
            ]
        )

    if role_name == "critic":
        return "\n\n".join(
            task_blocks
            + [
                "You are stage 4 of a fixed committee pipeline.",
                _tag_block("coordinator_plan", stage_outputs["coordinator_plan"]),
                _tag_block("researcher_notes", stage_outputs["researcher"]),
                _tag_block("analyst_draft", stage_outputs["analyst"]),
                (
                    "Find errors, weak assumptions, missing support, and overclaims. "
                    "Check whether the draft actually satisfies the deliverable and evaluation criteria. "
                    "Do not replace the whole answer unless the draft is fundamentally broken."
                ),
            ]
        )

    if role_name == "coordinator_finalize":
        return "\n\n".join(
            task_blocks
            + [
                "You are stage 5 of a fixed committee pipeline.",
                _tag_block("coordinator_plan", stage_outputs["coordinator_plan"]),
                _tag_block("researcher_notes", stage_outputs["researcher"]),
                _tag_block("analyst_draft", stage_outputs["analyst"]),
                _tag_block("critic_feedback", stage_outputs["critic"]),
                (
                    "Write the final answer after integrating the strongest parts of the draft "
                    "and resolving valid critique. Ensure the output matches the requested deliverable "
                    "and covers the evaluation criteria. Preserve only material residual risks."
                ),
            ]
        )

    raise ValueError(f"Unknown role: {role_name}")


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
        steps: list[dict[str, Any]] = []

        for role_name in ROLE_SEQUENCE:
            role_settings = self.config.resolve_role_settings(role_name)
            system_prompt = self.config.load_system_prompt(role_name)
            user_prompt = _build_role_prompt(role_name, task, stage_outputs)
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

            stage_outputs[role_name] = result.content
            steps.append(
                {
                    "role": role_name,
                    "model": result.model,
                    "temperature": role_settings["temperature"],
                    "max_tokens": role_settings["max_tokens"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": result.content,
                    "elapsed_sec": result.elapsed_sec,
                    "usage": result.usage,
                }
            )

        final_stage = stage_outputs["coordinator_finalize"]
        final_answer = _extract_section(final_stage, "Final Answer")
        summary = _build_trace_summary(steps)
        evaluation = evaluate_answer(task, final_answer)

        return {
            "run_id": run_id,
            "architecture": "committee_v1_explicit_coordination",
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
