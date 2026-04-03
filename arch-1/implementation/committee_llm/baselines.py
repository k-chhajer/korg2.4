from __future__ import annotations

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


def _tag_block(name: str, content: str) -> str:
    return f"<{name}>\n{content.strip()}\n</{name}>"


def _tag_list_block(name: str, items: list[str]) -> str:
    return _tag_block(name, "\n".join(f"- {item}" for item in items))


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

    return blocks


def _build_single_prompt(task: TaskSpec) -> str:
    instructions = [
        "You are a single-pass baseline system for a research benchmark.",
        "Solve the task directly in one response.",
        "If context is provided, rely on the provided context rather than unsupported external claims.",
    ]

    if task.choices:
        instructions.append(
            "If answer choices are provided, begin the response with `Answer: <label>` using exactly one listed label."
        )
    else:
        instructions.append("Begin with the shortest direct answer you can justify, then provide a brief explanation.")

    return "\n\n".join(_build_task_blocks(task) + instructions)


@dataclass(slots=True)
class SingleShotRunner:
    config: ExperimentConfig
    client: OpenAICompatibleChatClient
    temperature: float | None = None
    max_tokens: int | None = None

    def run_task(self, task: TaskSpec) -> dict[str, Any]:
        run_id = task.task_id or str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        run_start = time.perf_counter()
        user_prompt = _build_single_prompt(task)
        system_prompt = (
            "You are a careful baseline model for benchmark evaluation. "
            "Follow the task instructions exactly and avoid unnecessary verbosity."
        )

        result = self.client.chat(
            model=self.config.defaults.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if self.temperature is not None else self.config.defaults.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else self.config.defaults.max_tokens,
        )

        steps = [
            {
                "role": "single_direct",
                "model": result.model,
                "temperature": self.temperature if self.temperature is not None else self.config.defaults.temperature,
                "max_tokens": self.max_tokens if self.max_tokens is not None else self.config.defaults.max_tokens,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": result.content,
                "elapsed_sec": result.elapsed_sec,
                "usage": result.usage,
            }
        ]

        return {
            "run_id": run_id,
            "architecture": "single_direct_baseline",
            "config_name": self.config.name,
            "config_path": str(self.config.source_path),
            "started_at_utc": started_at,
            "total_elapsed_sec": time.perf_counter() - run_start,
            "input": task.prompt,
            "task": task.to_trace_payload(),
            "metadata": dict(task.metadata),
            "steps": steps,
            "final_stage": result.content,
            "final_answer": result.content.strip(),
            "evaluation": evaluate_answer(task, result.content.strip()),
            "summary": {
                "stage_count": 1,
                "per_role_elapsed_sec": {"single_direct": result.elapsed_sec},
                "per_role_usage": {"single_direct": result.usage if isinstance(result.usage, dict) else {}},
                "usage": aggregate_usage([result.usage]),
            },
        }
