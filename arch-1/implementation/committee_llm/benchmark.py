from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from committee_llm.baselines import SingleShotRunner
from committee_llm.client import OpenAICompatibleChatClient
from committee_llm.config import ExperimentConfig
from committee_llm.metrics import aggregate_usage, merge_numeric_dicts
from committee_llm.orchestrator import CommitteeRunner
from committee_llm.tasks import TaskSpec


def _load_tasks(tasks_path: str | Path, limit: int | None = None) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for line_number, raw_line in enumerate(Path(tasks_path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Task line {line_number} in {tasks_path} must be a JSON object.")
        tasks.append(TaskSpec.from_dict(item, default_task_id=f"task-{line_number:05d}"))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _aggregate_reference_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    overall_sums: dict[str, float] = {}
    overall_counts: dict[str, int] = {}
    by_benchmark: dict[str, dict[str, Any]] = {}

    for trace in traces:
        reference = trace.get("evaluation", {}).get("reference", {})
        metrics = reference.get("metrics", {})
        benchmark_name = str(reference.get("benchmark_name") or trace.get("task", {}).get("benchmark_name") or "unspecified")

        benchmark_bucket = by_benchmark.setdefault(
            benchmark_name,
            {"count": 0, "metric_sums": {}, "metric_counts": {}},
        )
        benchmark_bucket["count"] += 1

        if not isinstance(metrics, dict):
            continue

        for name, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue

            overall_sums[name] = overall_sums.get(name, 0.0) + float(value)
            overall_counts[name] = overall_counts.get(name, 0) + 1

            metric_sums = benchmark_bucket["metric_sums"]
            metric_counts = benchmark_bucket["metric_counts"]
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            metric_counts[name] = metric_counts.get(name, 0) + 1

    overall = {
        name: overall_sums[name] / overall_counts[name]
        for name in sorted(overall_sums)
        if overall_counts[name]
    }
    benchmark_summary = {}
    for benchmark_name, bucket in by_benchmark.items():
        benchmark_summary[benchmark_name] = {
            "count": bucket["count"],
            "metrics": {
                name: bucket["metric_sums"][name] / bucket["metric_counts"][name]
                for name in sorted(bucket["metric_sums"])
                if bucket["metric_counts"][name]
            },
        }

    return {"overall": overall, "by_benchmark": benchmark_summary}


def _build_system_summary(
    *,
    system_name: str,
    completed_runs: list[dict[str, Any]],
    failures: list[dict[str, str]],
    started_at_utc: str,
    wall_clock_elapsed_sec: float,
) -> dict[str, Any]:
    usage_by_role: dict[str, dict[str, Any]] = {}
    total_run_elapsed_sec = 0.0

    for item in completed_runs:
        trace = item["trace"]
        total_run_elapsed_sec += float(trace["total_elapsed_sec"])
        per_role_usage = trace.get("summary", {}).get("per_role_usage", {})
        if isinstance(per_role_usage, dict):
            for role_name, usage in per_role_usage.items():
                bucket = usage_by_role.setdefault(role_name, {})
                merge_numeric_dicts(bucket, usage if isinstance(usage, dict) else {})

    traces = [item["trace"] for item in completed_runs]
    aggregate_usage_totals = aggregate_usage([trace.get("summary", {}).get("usage") for trace in traces])
    reference_metrics = _aggregate_reference_metrics(traces)

    return {
        "system": system_name,
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_count": len(completed_runs),
        "failed_count": len(failures),
        "wall_clock_elapsed_sec": wall_clock_elapsed_sec,
        "aggregate": {
            "run_elapsed_sec": total_run_elapsed_sec,
            "mean_run_elapsed_sec": total_run_elapsed_sec / len(completed_runs) if completed_runs else None,
            "tasks_per_sec": len(completed_runs) / wall_clock_elapsed_sec if wall_clock_elapsed_sec > 0 else None,
            "usage": aggregate_usage_totals,
            "per_role_usage": usage_by_role,
            "reference_metrics": reference_metrics,
        },
        "runs": [
            {
                "task_id": item["task_id"],
                "output_path": str(item["output_path"]),
                "total_elapsed_sec": item["trace"]["total_elapsed_sec"],
                "reference_metrics": item["trace"].get("evaluation", {}).get("reference", {}).get("metrics", {}),
            }
            for item in completed_runs
        ],
        "failures": failures,
    }


def _build_comparison(system_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "committee" not in system_summaries or "single_direct" not in system_summaries:
        return {}

    committee = system_summaries["committee"]
    baseline = system_summaries["single_direct"]

    def _ratio(a: float | None, b: float | None) -> float | None:
        if a is None or b in (None, 0):
            return None
        return a / b

    comparison = {
        "committee_vs_single_direct": {
            "latency_ratio": _ratio(
                committee["aggregate"]["mean_run_elapsed_sec"],
                baseline["aggregate"]["mean_run_elapsed_sec"],
            ),
            "token_ratio": _ratio(
                committee["aggregate"]["usage"].get("total_tokens"),
                baseline["aggregate"]["usage"].get("total_tokens"),
            ),
            "metric_deltas": {},
        }
    }

    committee_metrics = committee["aggregate"]["reference_metrics"]["overall"]
    baseline_metrics = baseline["aggregate"]["reference_metrics"]["overall"]
    common_metrics = sorted(set(committee_metrics) & set(baseline_metrics))
    for metric in common_metrics:
        comparison["committee_vs_single_direct"]["metric_deltas"][metric] = (
            float(committee_metrics[metric]) - float(baseline_metrics[metric])
        )

    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research benchmarks and compare committee vs baseline systems.")
    parser.add_argument("--config", required=True, help="Path to an experiment config JSON file.")
    parser.add_argument("--tasks", required=True, help="Path to converted benchmark task JSONL.")
    parser.add_argument("--outdir", default="runs/benchmark", help="Output directory for benchmark traces.")
    parser.add_argument("--system", choices=["committee", "single_direct", "both"], default="both")
    parser.add_argument("--limit", type=int, help="Optional max number of tasks to run.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--single-temperature", type=float, help="Optional override for the single baseline temperature.")
    parser.add_argument("--single-max-tokens", type=int, help="Optional override for the single baseline max tokens.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.load(args.config)
    tasks = _load_tasks(args.tasks, limit=args.limit)
    client = OpenAICompatibleChatClient(
        api_base=config.provider.api_base,
        api_key_env=config.provider.api_key_env,
        timeout_sec=config.provider.timeout_sec,
    )

    systems: dict[str, Any] = {}
    if args.system in {"committee", "both"}:
        systems["committee"] = CommitteeRunner(config=config, client=client)
    if args.system in {"single_direct", "both"}:
        systems["single_direct"] = SingleShotRunner(
            config=config,
            client=client,
            temperature=args.single_temperature,
            max_tokens=args.single_max_tokens,
        )

    outdir = Path(args.outdir)
    system_summaries: dict[str, dict[str, Any]] = {}
    exit_code = 0

    for system_name, runner in systems.items():
        started_at_utc = datetime.now(timezone.utc).isoformat()
        system_start = time.perf_counter()
        completed_runs: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []

        for index, task in enumerate(tasks, start=1):
            task_id = str(task.task_id or f"task-{index:05d}")
            task.task_id = task_id
            try:
                trace = runner.run_task(task)
            except Exception as exc:
                failures.append({"task_id": task_id, "error": str(exc)})
                print(f"[{system_name} {index}/{len(tasks)}] failed {task_id}: {exc}", file=sys.stderr)
                exit_code = 1
                if not args.continue_on_error:
                    break
                continue

            output_path = outdir / system_name / f"{task_id}.json"
            _write_json(output_path, trace)
            completed_runs.append({"task_id": task_id, "output_path": output_path, "trace": trace})
            print(f"[{system_name} {index}/{len(tasks)}] wrote {output_path}", file=sys.stderr)

        system_summaries[system_name] = _build_system_summary(
            system_name=system_name,
            completed_runs=completed_runs,
            failures=failures,
            started_at_utc=started_at_utc,
            wall_clock_elapsed_sec=time.perf_counter() - system_start,
        )

    benchmark_summary = {
        "config_name": config.name,
        "config_path": str(config.source_path),
        "task_count": len(tasks),
        "systems": system_summaries,
        "comparison": _build_comparison(system_summaries),
    }
    summary_path = outdir / "summary.json"
    _write_json(summary_path, benchmark_summary)
    print(f"wrote {summary_path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
