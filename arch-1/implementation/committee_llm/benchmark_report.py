from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark summary at {path} must be a JSON object.")
    return payload


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text
    return str(value)


def _escape_cell(value: Any) -> str:
    return _format_number(value).replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def _make_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_row, separator, *body])


def _iter_systems(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    systems = summary.get("systems", {})
    if not isinstance(systems, dict):
        return []
    return [(name, payload) for name, payload in sorted(systems.items()) if isinstance(payload, dict)]


def _overall_metric_names(summary: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for _, system_payload in _iter_systems(summary):
        metrics = system_payload.get("aggregate", {}).get("reference_metrics", {}).get("overall", {})
        if isinstance(metrics, dict):
            names.update(str(name) for name in metrics)
    return sorted(names)


def _benchmark_metric_names(summary: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for _, system_payload in _iter_systems(summary):
        by_benchmark = system_payload.get("aggregate", {}).get("reference_metrics", {}).get("by_benchmark", {})
        if not isinstance(by_benchmark, dict):
            continue
        for benchmark_payload in by_benchmark.values():
            if not isinstance(benchmark_payload, dict):
                continue
            metrics = benchmark_payload.get("metrics", {})
            if isinstance(metrics, dict):
                names.update(str(name) for name in metrics)
    return sorted(names)


def _comparison_metric_names(summary: dict[str, Any]) -> list[str]:
    comparison = summary.get("comparison", {})
    if not isinstance(comparison, dict):
        return []

    names: set[str] = set()
    for payload in comparison.values():
        if not isinstance(payload, dict):
            continue
        metric_deltas = payload.get("metric_deltas", {})
        if isinstance(metric_deltas, dict):
            names.update(str(name) for name in metric_deltas)
    return sorted(names)


def render_markdown(summary: dict[str, Any], summary_path: Path) -> str:
    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "# Benchmark Results",
        "",
        f"- Summary source: `{summary_path}`",
        f"- Generated at (UTC): `{generated_at_utc}`",
        f"- Config: `{summary.get('config_name', 'N/A')}`",
        f"- Config path: `{summary.get('config_path', 'N/A')}`",
        f"- Task count: `{_format_number(summary.get('task_count'))}`",
        "",
        "## System Overview",
        "",
    ]

    overview_rows: list[list[Any]] = []
    for system_name, system_payload in _iter_systems(summary):
        aggregate = system_payload.get("aggregate", {})
        usage = aggregate.get("usage", {}) if isinstance(aggregate, dict) else {}
        overview_rows.append(
            [
                system_name,
                system_payload.get("completed_count"),
                system_payload.get("failed_count"),
                system_payload.get("wall_clock_elapsed_sec"),
                aggregate.get("mean_run_elapsed_sec") if isinstance(aggregate, dict) else None,
                aggregate.get("tasks_per_sec") if isinstance(aggregate, dict) else None,
                usage.get("total_tokens") if isinstance(usage, dict) else None,
            ]
        )

    if overview_rows:
        lines.extend(
            [
                _make_table(
                    [
                        "System",
                        "Completed",
                        "Failed",
                        "Wall Clock (s)",
                        "Mean Run (s)",
                        "Tasks/s",
                        "Total Tokens",
                    ],
                    overview_rows,
                ),
                "",
            ]
        )
    else:
        lines.extend(["_No systems were recorded in this summary._", ""])

    overall_metric_names = _overall_metric_names(summary)
    lines.extend(["## Overall Metrics", ""])
    if overall_metric_names:
        overall_rows: list[list[Any]] = []
        for system_name, system_payload in _iter_systems(summary):
            metrics = system_payload.get("aggregate", {}).get("reference_metrics", {}).get("overall", {})
            overall_rows.append([system_name, *[metrics.get(name) if isinstance(metrics, dict) else None for name in overall_metric_names]])
        lines.extend([_make_table(["System", *overall_metric_names], overall_rows), ""])
    else:
        lines.extend(["_No completed benchmark metrics were recorded in this summary._", ""])

    benchmark_metric_names = _benchmark_metric_names(summary)
    lines.extend(["## Per-Benchmark Metrics", ""])
    benchmark_rows: list[list[Any]] = []
    for system_name, system_payload in _iter_systems(summary):
        by_benchmark = system_payload.get("aggregate", {}).get("reference_metrics", {}).get("by_benchmark", {})
        if not isinstance(by_benchmark, dict):
            continue
        for benchmark_name, benchmark_payload in sorted(by_benchmark.items()):
            if not isinstance(benchmark_payload, dict):
                continue
            metrics = benchmark_payload.get("metrics", {})
            benchmark_rows.append(
                [
                    system_name,
                    benchmark_name,
                    benchmark_payload.get("count"),
                    *[metrics.get(name) if isinstance(metrics, dict) else None for name in benchmark_metric_names],
                ]
            )

    if benchmark_rows:
        lines.extend(
            [
                _make_table(["System", "Benchmark", "Count", *benchmark_metric_names], benchmark_rows),
                "",
            ]
        )
    else:
        lines.extend(["_No per-benchmark metrics were recorded in this summary._", ""])

    comparison_metric_names = _comparison_metric_names(summary)
    comparison = summary.get("comparison", {})
    lines.extend(["## Comparison", ""])
    comparison_rows: list[list[Any]] = []
    if isinstance(comparison, dict):
        for comparison_name, payload in sorted(comparison.items()):
            if not isinstance(payload, dict):
                continue
            metric_deltas = payload.get("metric_deltas", {})
            comparison_rows.append(
                [
                    comparison_name,
                    payload.get("latency_ratio"),
                    payload.get("token_ratio"),
                    *[
                        metric_deltas.get(name) if isinstance(metric_deltas, dict) else None
                        for name in comparison_metric_names
                    ],
                ]
            )

    if comparison_rows:
        lines.extend(
            [
                _make_table(
                    ["Comparison", "Latency Ratio", "Token Ratio", *[f"{name} Delta" for name in comparison_metric_names]],
                    comparison_rows,
                ),
                "",
            ]
        )
    else:
        lines.extend(["_No system comparison block was recorded in this summary._", ""])

    failure_rows: list[list[Any]] = []
    for system_name, system_payload in _iter_systems(summary):
        failures = system_payload.get("failures", [])
        if not isinstance(failures, list):
            continue
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            failure_rows.append([system_name, failure.get("task_id"), failure.get("error")])

    lines.extend(["## Failures", ""])
    if failure_rows:
        lines.extend([_make_table(["System", "Task ID", "Error"], failure_rows), ""])
    else:
        lines.extend(["_No failures were recorded in this summary._", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a benchmark summary.json file into a Markdown report.")
    parser.add_argument("--summary", required=True, help="Path to a benchmark summary.json file.")
    parser.add_argument("--output", help="Optional output Markdown path. Defaults to the summary path with a .md suffix.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary)
    output_path = Path(args.output) if args.output else summary_path.with_suffix(".md")
    markdown = render_markdown(_load_summary(summary_path), summary_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
