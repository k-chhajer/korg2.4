#!/usr/bin/env python3
"""Generate evaluation comparison graphs from summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def short_label(path: Path, system: str) -> str:
    parts = path.parts
    try:
        idx = parts.index("runs")
        run_parts = parts[idx + 1 : -1]
    except ValueError:
        run_parts = parts[-3:-1]
    run = "/".join(run_parts)
    run = run.replace("gpqa_diamond_modal_committee_198_20260405/", "")
    run = run.replace("hotpotqa_", "hotpot_")
    run = run.replace("gpqa_", "gpqa_")
    return f"{run}\n{system}"


def wrap_labels(labels: list[str], width: int = 30) -> list[str]:
    return ["\n".join(textwrap.wrap(label, width=width)) for label in labels]


def load_rows(runs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(runs_dir.rglob("summary.json")):
        with summary_path.open() as f:
            data = json.load(f)
        for system, record in data.get("systems", {}).items():
            aggregate = record.get("aggregate", {})
            reference = aggregate.get("reference_metrics", {}).get("overall", {})
            behavior = aggregate.get("behavior_metrics", {})
            usage = aggregate.get("usage", {})
            completed = record.get("completed_count", 0) or 0
            failed = record.get("failed_count", 0) or 0
            total = completed + failed
            primary_metric_name = "choice_accuracy" if "choice_accuracy" in reference else "answer_f1"
            primary_metric = reference.get(primary_metric_name)
            benchmark = next(
                iter(aggregate.get("reference_metrics", {}).get("by_benchmark", {"unknown": {}}).keys()),
                "unknown",
            )
            rows.append(
                {
                    "summary_path": str(summary_path),
                    "label": short_label(summary_path, system),
                    "run": str(summary_path.parent.relative_to(runs_dir)),
                    "system": system,
                    "benchmark": benchmark,
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "failure_rate": failed / total if total else 0.0,
                    "started_at_utc": record.get("started_at_utc") or "",
                    "mean_elapsed_sec": aggregate.get("mean_run_elapsed_sec"),
                    "tasks_per_sec": aggregate.get("tasks_per_sec"),
                    "total_tokens": usage.get("total_tokens"),
                    "tokens_per_task": (usage.get("total_tokens") / completed)
                    if completed and usage.get("total_tokens") is not None
                    else None,
                    "model_call_count": aggregate.get("model_call_count"),
                    "mean_model_call_count": aggregate.get("mean_model_call_count"),
                    "primary_metric_name": primary_metric_name,
                    "primary_metric": primary_metric,
                    "answer_f1": reference.get("answer_f1"),
                    "answer_em": reference.get("answer_em"),
                    "choice_accuracy": reference.get("choice_accuracy"),
                    "supporting_fact_title_recall": reference.get("supporting_fact_title_recall"),
                    "supporting_fact_title_precision": reference.get("supporting_fact_title_precision"),
                    "decomposition_rate": behavior.get("decomposition_rate"),
                    "evidence_grounding_rate": behavior.get("evidence_grounding_rate"),
                    "premature_finalization_rate": behavior.get("premature_finalization_rate"),
                    "revision_after_critique_rate": behavior.get("revision_after_critique_rate"),
                    "unsupported_claim_rate": behavior.get("unsupported_claim_rate"),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "run",
        "system",
        "benchmark",
        "completed",
        "failed",
        "failure_rate",
        "started_at_utc",
        "mean_elapsed_sec",
        "tasks_per_sec",
        "tokens_per_task",
        "primary_metric_name",
        "primary_metric",
        "answer_f1",
        "answer_em",
        "choice_accuracy",
        "supporting_fact_title_recall",
        "supporting_fact_title_precision",
        "decomposition_rate",
        "evidence_grounding_rate",
        "premature_finalization_rate",
        "revision_after_critique_rate",
        "unsupported_claim_rate",
        "summary_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def select_metric_rows(rows: list[dict[str, object]], metric: str, benchmark: str) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row.get(metric) is not None
        and (
            str(row.get("benchmark")) == benchmark
            or benchmark in str(row.get("run", ""))
            or (benchmark == "hotpotqa_distractor" and "hotpot" in str(row.get("run", "")))
        )
    ]
    selected.sort(key=lambda row: (str(row.get("started_at_utc")), str(row.get("run"))))
    return selected


def bar_metric(rows: list[dict[str, object]], metric: str, title: str, ylabel: str, path: Path) -> None:
    if not rows:
        return
    labels = wrap_labels([str(row["label"]) for row in rows])
    values = [pct(float(row[metric])) for row in rows]
    fig_h = max(5.0, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.barh(labels, values, color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_xlim(0, max(100, math.ceil(max(values) / 10) * 10))
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for idx, value in enumerate(values):
        ax.text(value + 0.8, idx, f"{value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def latency_scatter(rows: list[dict[str, object]], path: Path) -> None:
    points = [
        row
        for row in rows
        if row.get("primary_metric") is not None
        and row.get("mean_elapsed_sec") is not None
        and row.get("completed", 0)
    ]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        "gpqa": "#2f6f9f",
        "hotpotqa_distractor": "#bc6c25",
        "unknown": "#6a994e",
    }
    for row in points:
        benchmark = str(row.get("benchmark"))
        ax.scatter(
            float(row["mean_elapsed_sec"]),
            pct(float(row["primary_metric"])),
            s=max(35, min(380, float(row["completed"]) * 5)),
            alpha=0.72,
            color=colors.get(benchmark, "#6a994e"),
            edgecolor="white",
            linewidth=0.7,
        )
        ax.annotate(
            str(row["run"]).replace("gpqa_diamond_modal_committee_198_20260405/", ""),
            (float(row["mean_elapsed_sec"]), pct(float(row["primary_metric"]))),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.82,
        )
    ax.set_title("Quality vs Mean Runtime")
    ax.set_xlabel("Mean runtime per completed task (seconds)")
    ax.set_ylabel("Primary quality metric (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def token_bar(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row.get("tokens_per_task") is not None and row.get("completed", 0)]
    selected.sort(key=lambda row: float(row["tokens_per_task"]))
    selected = selected[-30:]
    if not selected:
        return
    labels = wrap_labels([str(row["label"]) for row in selected])
    values = [float(row["tokens_per_task"]) for row in selected]
    fig_h = max(5.0, 0.42 * len(selected))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.barh(labels, values, color="#6a994e")
    ax.set_title("Token Cost Per Completed Task")
    ax.set_xlabel("Total tokens / completed task")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def behavior_bars(rows: list[dict[str, object]], path: Path) -> None:
    metrics = [
        "decomposition_rate",
        "evidence_grounding_rate",
        "premature_finalization_rate",
        "revision_after_critique_rate",
        "unsupported_claim_rate",
    ]
    selected = [
        row
        for row in rows
        if any(row.get(metric) is not None for metric in metrics)
        and str(row.get("system")) == "committee"
        and row.get("completed", 0)
    ]
    selected.sort(key=lambda row: (str(row.get("benchmark")), str(row.get("run"))))
    if not selected:
        return
    x = list(range(len(selected)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2f6f9f", "#6a994e", "#bc4749", "#735d78", "#5f6c7b"]
    for offset, metric in enumerate(metrics):
        values = [pct(float(row.get(metric) or 0.0)) for row in selected]
        positions = [item + (offset - 2) * width for item in x]
        ax.bar(positions, values, width=width, label=metric.replace("_", " "), color=colors[offset])
    ax.set_title("Committee Behavior Metrics")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels([str(row["run"]) for row in selected], width=22), rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def failure_bar(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row.get("total", 0)]
    selected.sort(key=lambda row: (float(row["failure_rate"]), str(row["run"])))
    selected = selected[-30:]
    if not selected:
        return
    labels = wrap_labels([str(row["label"]) for row in selected])
    values = [pct(float(row["failure_rate"])) for row in selected]
    fig_h = max(5.0, 0.42 * len(selected))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.barh(labels, values, color="#bc4749")
    ax.set_title("Failure Rate By Run")
    ax.set_xlabel("Failed tasks / attempted tasks (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="arch-1/evals/runs")
    parser.add_argument("--outdir", default="arch-1/evals/reports/graphs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(runs_dir)
    write_csv(rows, outdir / "eval_graph_data.csv")
    bar_metric(
        select_metric_rows(rows, "choice_accuracy", "gpqa"),
        "choice_accuracy",
        "GPQA Accuracy By Run",
        "Choice accuracy (%)",
        outdir / "gpqa_accuracy_by_run.png",
    )
    bar_metric(
        select_metric_rows(rows, "answer_f1", "hotpotqa_distractor"),
        "answer_f1",
        "HotpotQA Answer F1 By Run",
        "Answer F1 (%)",
        outdir / "hotpotqa_answer_f1_by_run.png",
    )
    latency_scatter(rows, outdir / "quality_vs_runtime.png")
    token_bar(rows, outdir / "tokens_per_completed_task.png")
    behavior_bars(rows, outdir / "committee_behavior_metrics.png")
    failure_bar(rows, outdir / "failure_rate_by_run.png")
    print(f"Wrote {len(rows)} rows and graphs to {outdir}")


if __name__ == "__main__":
    main()
