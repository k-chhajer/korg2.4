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
from committee_llm.orchestrator import _build_task_blocks
from committee_llm.tasks import TaskSpec


def _single_summary(role_name: str, result: Any) -> dict[str, Any]:
    usage = result.usage if isinstance(result.usage, dict) else {}
    return {
        "stage_count": 1,
        "model_call_count": 1,
        "per_role_elapsed_sec": {role_name: result.elapsed_sec},
        "per_role_usage": {role_name: usage},
        "usage": aggregate_usage([usage]),
    }


def _trace_payload(
    *,
    run_id: str,
    architecture: str,
    config: ExperimentConfig,
    started_at: str,
    run_start: float,
    task: TaskSpec,
    steps: list[dict[str, Any]],
    final_stage: str,
    final_answer: str,
) -> dict[str, Any]:
    per_role_usage = [step.get("usage") for step in steps]
    summary = {
        "stage_count": len(steps),
        "model_call_count": len(steps),
        "per_role_elapsed_sec": {str(step["role"]): float(step["elapsed_sec"]) for step in steps},
        "per_role_usage": {
            str(step["role"]): step["usage"] if isinstance(step.get("usage"), dict) else {}
            for step in steps
        },
        "usage": aggregate_usage(per_role_usage),
    }

    return {
        "run_id": run_id,
        "architecture": architecture,
        "config_name": config.name,
        "config_path": str(config.source_path),
        "started_at_utc": started_at,
        "total_elapsed_sec": time.perf_counter() - run_start,
        "input": task.prompt,
        "task": task.to_trace_payload(),
        "metadata": dict(task.metadata),
        "steps": steps,
        "final_stage": final_stage,
        "final_answer": final_answer,
        "evaluation": evaluate_answer(task, final_answer),
        "summary": summary,
    }


def _base_system_prompt() -> str:
    return (
        "You are a careful single-model baseline for controlled research evaluation. "
        "Follow the task exactly, rely on the provided context when available, and avoid unsupported claims."
    )


def _task_prompt(task: TaskSpec) -> str:
    return "\n\n".join(_build_task_blocks(task).values())


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
        user_prompt = "\n\n".join(
            [
                _task_prompt(task),
                "Solve the task directly in one response.",
                "Begin with the shortest direct answer you can justify, then briefly justify it.",
            ]
        )

        result = self.client.chat(
            model=self.config.defaults.model,
            messages=[
                {"role": "system", "content": _base_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if self.temperature is not None else self.config.defaults.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else self.config.defaults.max_tokens,
            extra_body=self.config.defaults.extra_body,
        )

        step = {
            "role": "single_shot",
            "model": result.model,
            "temperature": self.temperature if self.temperature is not None else self.config.defaults.temperature,
            "max_tokens": self.max_tokens if self.max_tokens is not None else self.config.defaults.max_tokens,
            "extra_body": self.config.defaults.extra_body,
            "system_prompt": _base_system_prompt(),
            "user_prompt": user_prompt,
            "response": result.content,
            "elapsed_sec": result.elapsed_sec,
            "usage": result.usage,
        }
        return _trace_payload(
            run_id=run_id,
            architecture="single_shot_baseline",
            config=self.config,
            started_at=started_at,
            run_start=run_start,
            task=task,
            steps=[step],
            final_stage=result.content,
            final_answer=result.content.strip(),
        )


@dataclass(slots=True)
class SingleModelMultiTurnRunner:
    config: ExperimentConfig
    client: OpenAICompatibleChatClient

    def run_task(self, task: TaskSpec) -> dict[str, Any]:
        run_id = task.task_id or str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        run_start = time.perf_counter()
        messages = [
            {"role": "system", "content": _base_system_prompt()},
            {"role": "user", "content": _task_prompt(task)},
        ]
        steps: list[dict[str, Any]] = []

        turn_specs = [
            (
                "single_multiturn_plan",
                "Decompose the task into a short plan. Keep it concise and operational.",
            ),
            (
                "single_multiturn_evidence",
                "Gather the most relevant evidence, assumptions, and unresolved issues from the provided materials.",
            ),
            (
                "single_multiturn_answer",
                "Produce the final answer using the plan and evidence above. Keep only material caveats.",
            ),
        ]

        final_answer = ""
        for role_name, instruction in turn_specs:
            messages.append({"role": "user", "content": instruction})
            result = self.client.chat(
                model=self.config.defaults.model,
                messages=messages,
                temperature=self.config.defaults.temperature,
                max_tokens=self.config.defaults.max_tokens,
                extra_body=self.config.defaults.extra_body,
            )
            messages.append({"role": "assistant", "content": result.content})
            steps.append(
                {
                    "role": role_name,
                    "model": result.model,
                    "temperature": self.config.defaults.temperature,
                    "max_tokens": self.config.defaults.max_tokens,
                    "extra_body": self.config.defaults.extra_body,
                    "system_prompt": _base_system_prompt(),
                    "user_prompt": instruction,
                    "response": result.content,
                    "elapsed_sec": result.elapsed_sec,
                    "usage": result.usage,
                }
            )
            final_answer = result.content.strip()

        return _trace_payload(
            run_id=run_id,
            architecture="single_model_multiturn_baseline",
            config=self.config,
            started_at=started_at,
            run_start=run_start,
            task=task,
            steps=steps,
            final_stage=final_answer,
            final_answer=final_answer,
        )


@dataclass(slots=True)
class SingleModelStructuredRunner:
    config: ExperimentConfig
    client: OpenAICompatibleChatClient

    def run_task(self, task: TaskSpec) -> dict[str, Any]:
        run_id = task.task_id or str(uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        run_start = time.perf_counter()
        user_prompt = "\n\n".join(
            [
                _task_prompt(task),
                "Solve the task using this internal structure in a single response:",
                "1. Decomposition",
                "2. Evidence",
                "3. Tentative Answer",
                "4. Self-Critique",
                "5. Final Answer",
                "Return all five headers exactly.",
            ]
        )

        result = self.client.chat(
            model=self.config.defaults.model,
            messages=[
                {"role": "system", "content": _base_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.defaults.temperature,
            max_tokens=self.config.defaults.max_tokens,
            extra_body=self.config.defaults.extra_body,
        )

        final_answer = result.content.split("Final Answer:", 1)[-1].strip() if "Final Answer:" in result.content else result.content.strip()
        step = {
            "role": "single_structured",
            "model": result.model,
            "temperature": self.config.defaults.temperature,
            "max_tokens": self.config.defaults.max_tokens,
            "extra_body": self.config.defaults.extra_body,
            "system_prompt": _base_system_prompt(),
            "user_prompt": user_prompt,
            "response": result.content,
            "elapsed_sec": result.elapsed_sec,
            "usage": result.usage,
        }
        return _trace_payload(
            run_id=run_id,
            architecture="single_model_structured_baseline",
            config=self.config,
            started_at=started_at,
            run_start=run_start,
            task=task,
            steps=[step],
            final_stage=result.content,
            final_answer=final_answer,
        )
