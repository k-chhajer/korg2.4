from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig
from committee_llm.metrics import aggregate_usage, merge_numeric_dicts
from committee_llm.orchestrator import CommitteeRunner
from committee_llm.tasks import TaskSpec


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    raise ValueError("Either --prompt or --prompt-file must be provided.")


def _load_task(task_path: str | Path) -> TaskSpec:
    payload = json.loads(Path(task_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Task file {task_path} must contain a JSON object.")
    return TaskSpec.from_dict(payload)


def _load_tasks(tasks_path: str | Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for line_number, raw_line in enumerate(Path(tasks_path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Task line {line_number} in {tasks_path} must be a JSON object.")
        tasks.append(TaskSpec.from_dict(item, default_task_id=f"task-{line_number:03d}"))
    return tasks


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_batch_summary(
    *,
    config: ExperimentConfig,
    requested_task_count: int,
    completed_runs: list[dict[str, Any]],
    failures: list[dict[str, str]],
    started_at_utc: str,
    wall_clock_elapsed_sec: float,
    continue_on_error: bool,
) -> dict[str, Any]:
    run_records: list[dict[str, Any]] = []
    usage_totals_by_role: dict[str, dict[str, Any]] = {}
    elapsed_totals_by_role: dict[str, float] = {}
    total_run_elapsed_sec = 0.0
    evaluation_scores: list[float] = []
    evaluation_passed_count = 0
    evaluation_failed_count = 0
    domain_summary: dict[str, dict[str, Any]] = {}

    for item in completed_runs:
        trace = item["trace"]
        trace_summary = trace.get("summary", {})
        run_usage = trace_summary.get("usage", {})
        per_role_usage = trace_summary.get("per_role_usage", {})
        per_role_elapsed = trace_summary.get("per_role_elapsed_sec", {})
        evaluation = trace.get("evaluation", {})
        task_payload = trace.get("task", {})
        domain = str(task_payload.get("domain") or "unspecified")

        run_records.append(
            {
                "task_id": item["task_id"],
                "run_id": trace["run_id"],
                "output_path": str(item["output_path"]),
                "domain": domain,
                "total_elapsed_sec": trace["total_elapsed_sec"],
                "usage": run_usage if isinstance(run_usage, dict) else {},
                "evaluation": {
                    "passed": evaluation.get("passed"),
                    "score": evaluation.get("score"),
                },
            }
        )

        total_run_elapsed_sec += float(trace["total_elapsed_sec"])
        domain_bucket = domain_summary.setdefault(
            domain,
            {"completed_count": 0, "scored_count": 0, "passed_count": 0, "scores": []},
        )
        domain_bucket["completed_count"] += 1

        if isinstance(per_role_usage, dict):
            for role_name, usage in per_role_usage.items():
                bucket = usage_totals_by_role.setdefault(role_name, {})
                merge_numeric_dicts(bucket, usage if isinstance(usage, dict) else {})

        if isinstance(per_role_elapsed, dict):
            for role_name, elapsed_sec in per_role_elapsed.items():
                if isinstance(elapsed_sec, (int, float)) and not isinstance(elapsed_sec, bool):
                    elapsed_totals_by_role[role_name] = elapsed_totals_by_role.get(role_name, 0.0) + float(
                        elapsed_sec
                    )

        score = evaluation.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            score_value = float(score)
            evaluation_scores.append(score_value)
            domain_bucket["scored_count"] += 1
            domain_bucket["scores"].append(score_value)

        passed = evaluation.get("passed")
        if passed is True:
            evaluation_passed_count += 1
            domain_bucket["passed_count"] += 1
        elif passed is False:
            evaluation_failed_count += 1

    attempted_task_count = len(completed_runs) + len(failures)

    if failures:
        status = "partial_failure" if continue_on_error and attempted_task_count == requested_task_count else "failed"
    else:
        status = "completed"

    evaluation_by_domain: dict[str, dict[str, Any]] = {}
    for domain, bucket in domain_summary.items():
        scores = [float(score) for score in bucket.pop("scores")]
        evaluation_by_domain[domain] = {
            "completed_count": bucket["completed_count"],
            "scored_count": bucket["scored_count"],
            "passed_count": bucket["passed_count"],
            "mean_score": sum(scores) / len(scores) if scores else None,
        }

    return {
        "status": status,
        "config_name": config.name,
        "config_path": str(config.source_path),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_task_count": requested_task_count,
        "attempted_task_count": attempted_task_count,
        "completed_count": len(completed_runs),
        "failed_count": len(failures),
        "continue_on_error": continue_on_error,
        "wall_clock_elapsed_sec": wall_clock_elapsed_sec,
        "aggregate": {
            "run_elapsed_sec": total_run_elapsed_sec,
            "usage": aggregate_usage(
                [
                    item["trace"].get("summary", {}).get("usage")
                    for item in completed_runs
                ]
            ),
            "per_role_elapsed_sec": elapsed_totals_by_role,
            "per_role_usage": usage_totals_by_role,
            "evaluation": {
                "scored_count": len(evaluation_scores),
                "passed_count": evaluation_passed_count,
                "failed_count": evaluation_failed_count,
                "mean_score": sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else None,
                "by_domain": evaluation_by_domain,
            },
        },
        "runs": run_records,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a role-specialized committee LLM pipeline on one prompt or a JSONL batch."
    )
    parser.add_argument("--config", required=True, help="Path to an experiment config JSON file.")
    parser.add_argument("--prompt", help="Single prompt to run.")
    parser.add_argument("--prompt-file", help="Path to a text file containing one prompt.")
    parser.add_argument("--task", help="Path to a single structured research-task JSON file.")
    parser.add_argument("--task-id", help="Optional explicit run id for single-prompt mode.")
    parser.add_argument("--output", help="Optional JSON path for saving a single-run trace.")
    parser.add_argument("--tasks", help="Path to JSONL tasks for batch mode.")
    parser.add_argument("--outdir", help="Output directory for batch traces.", default="runs")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In batch mode, keep running remaining tasks and record failures in summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    single_prompt_mode = bool(args.prompt or args.prompt_file or args.task)
    batch_mode = bool(args.tasks)

    if single_prompt_mode == batch_mode:
        parser.error("Choose exactly one mode: single prompt (--prompt or --prompt-file) or batch (--tasks).")

    config = ExperimentConfig.load(args.config)
    client = OpenAICompatibleChatClient(
        api_base=config.provider.api_base,
        api_key_env=config.provider.api_key_env,
        timeout_sec=config.provider.timeout_sec,
    )
    runner = CommitteeRunner(config=config, client=client)

    if single_prompt_mode:
        if args.task:
            task = _load_task(args.task)
            if args.task_id:
                task.task_id = args.task_id
        else:
            task = TaskSpec.from_prompt(_read_prompt(args), task_id=args.task_id)
        trace = runner.run_task(task)
        if args.output:
            _write_json(Path(args.output), trace)
        print(trace["final_answer"])
        return 0

    outdir = Path(args.outdir)
    tasks = _load_tasks(args.tasks)
    batch_started_at = datetime.now(timezone.utc).isoformat()
    batch_start = time.perf_counter()
    completed_runs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, task in enumerate(tasks, start=1):
        task_id = str(task.task_id or f"task-{index:03d}")
        task.task_id = task_id
        try:
            trace = runner.run_task(task)
        except Exception as exc:
            failures.append({"task_id": task_id, "error": str(exc)})
            print(f"[{index}/{len(tasks)}] failed {task_id}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break
            continue

        output_path = outdir / f"{task_id}.json"
        _write_json(output_path, trace)
        completed_runs.append({"task_id": task_id, "output_path": output_path, "trace": trace})
        print(f"[{index}/{len(tasks)}] wrote {output_path}", file=sys.stderr)

    summary = _build_batch_summary(
        config=config,
        requested_task_count=len(tasks),
        completed_runs=completed_runs,
        failures=failures,
        started_at_utc=batch_started_at,
        wall_clock_elapsed_sec=time.perf_counter() - batch_start,
        continue_on_error=bool(args.continue_on_error),
    )
    summary_path = outdir / "summary.json"
    _write_json(summary_path, summary)
    print(f"wrote {summary_path}", file=sys.stderr)

    return 1 if failures else 0
