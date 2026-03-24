from __future__ import annotations

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

from src.orchestrator.pipeline import MultiAgentOrchestrator
from src.orchestrator.types import RunRecord


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def keyword_score(text: str, expected_keywords: list[str]) -> float:
    text_l = text.lower()
    if not expected_keywords:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_l)
    return hits / float(len(expected_keywords))


def role_distinctiveness(role_outputs: dict[str, str]) -> float:
    if len(role_outputs) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in combinations(sorted(role_outputs.keys()), 2):
        l_tokens = set(role_outputs[left].lower().split())
        r_tokens = set(role_outputs[right].lower().split())
        union = l_tokens | r_tokens
        inter = l_tokens & r_tokens
        if not union:
            distances.append(0.0)
            continue
        distances.append(1.0 - (len(inter) / len(union)))
    return sum(distances) / float(len(distances))


def run_eval(
    orchestrator: MultiAgentOrchestrator,
    tasks: list[dict[str, Any]],
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    per_task: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_id = str(task["task_id"])
        query = str(task["query"])
        expected_keywords = [str(x) for x in task.get("expected_keywords", [])]
        record: RunRecord = orchestrator.run(
            run_id=run_id,
            task_id=task_id,
            user_query=query,
            seed=seed + idx,
        )
        kw_score = keyword_score(record.final_answer, expected_keywords)
        distinct = role_distinctiveness(record.role_outputs)
        per_task.append(
            {
                "task_id": task_id,
                "keyword_score": kw_score,
                "role_distinctiveness": distinct,
                "critique_acceptance_rate": record.metrics["critique_acceptance_rate"],
                "final_answer": record.final_answer,
                "run_record": {
                    **asdict(record),
                    "events": [asdict(event) for event in record.events],
                },
            }
        )

    task_count = len(per_task) if per_task else 1
    report = {
        "run_id": run_id,
        "task_count": len(per_task),
        "mean_keyword_score": sum(t["keyword_score"] for t in per_task) / task_count,
        "mean_role_distinctiveness": sum(t["role_distinctiveness"] for t in per_task)
        / task_count,
        "mean_critique_acceptance_rate": sum(
            t["critique_acceptance_rate"] for t in per_task
        )
        / task_count,
        "per_task": per_task,
    }
    return report

