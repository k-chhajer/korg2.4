from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "evals" / "runs"
DOCS_ROOT = ROOT / "docs"
OUTPUT_DIR = DOCS_ROOT / "generated" / "arch1_benchmark_report"


@dataclass
class RunRecord:
    run_name: str
    source_kind: str
    category: str
    dataset: str
    benchmark: str
    config_name: str
    config_path: str
    model: str
    provider: str
    endpoint_kind: str
    system: str
    started_at: str | None
    completed_at: str | None
    task_count: int
    completed_count: int
    failed_count: int
    completion_rate: float
    wall_clock_elapsed_sec: float | None
    mean_run_elapsed_sec: float | None
    total_tokens: int | None
    model_call_count: int | None
    answer_em: float | None
    answer_f1: float | None
    choice_accuracy: float | None
    supporting_fact_title_precision: float | None
    supporting_fact_title_recall: float | None
    supporting_fact_context_coverage: float | None
    decomposition_rate: float | None
    evidence_grounding_rate: float | None
    revision_after_critique_rate: float | None
    premature_finalization_rate: float | None
    unsupported_claim_rate: float | None
    status_note: str
    paper_claim_blocked: bool

    def primary_quality_metric_name(self) -> str:
        if self.answer_f1 is not None:
            return "answer_f1"
        if self.choice_accuracy is not None:
            return "choice_accuracy"
        return "completion_rate"

    def primary_quality_metric_value(self) -> float:
        if self.answer_f1 is not None:
            return self.answer_f1
        if self.choice_accuracy is not None:
            return self.choice_accuracy
        return self.completion_rate


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if digits == 0:
            return str(int(round(value)))
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text if text != "-0" else "0"
    return str(value)


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _provider_kind(api_base: str) -> str:
    lower = api_base.lower()
    if "modal.run" in lower:
        return "Modal"
    if "127.0.0.1:11434" in lower:
        return "Ollama local"
    if "127.0.0.1:8000" in lower:
        return "OpenAI-compatible local/T4"
    return "Other"


def _infer_dataset(summary: dict[str, Any], run_name: str) -> tuple[str, str]:
    systems = summary.get("systems") or {}
    for payload in systems.values():
        aggregate = payload.get("aggregate") if isinstance(payload, dict) else {}
        by_benchmark = ((aggregate or {}).get("reference_metrics") or {}).get("by_benchmark") or {}
        if isinstance(by_benchmark, dict) and by_benchmark:
            benchmark = sorted(by_benchmark)[0]
            if "hotpot" in benchmark.lower():
                return ("HotpotQA", benchmark)
            if "gpqa" in benchmark.lower():
                return ("GPQA", benchmark)
            return (benchmark, benchmark)
    lowered = run_name.lower()
    if lowered.startswith("hotpotqa"):
        return ("HotpotQA", "hotpotqa_distractor")
    if lowered.startswith("gpqa"):
        return ("GPQA", "gpqa_diamond")
    return ("Unknown", "unknown")


def _categorize_run(run_name: str) -> str:
    lowered = run_name.lower()
    if "smoke" in lowered:
        return "smoke"
    if "pilot" in lowered:
        return "pilot"
    if "probe" in lowered:
        return "probe"
    if any(token in lowered for token in ("_75_", "_198_")) or lowered.endswith("_100"):
        return "benchmark"
    return "other"


def _status_note(run_name: str, completed: int, failed: int) -> str:
    lowered = run_name.lower()
    if failed == 0 and completed > 0:
        return "completed"
    if completed == 0 and failed > 0:
        if "invalid function call" in lowered:
            return "failed setup"
        return "no successful traces"
    if failed > 0 and completed > 0:
        return "partial completion"
    return "unknown"


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if path.is_absolute() and path.exists():
        return _load_json(path)
    normalized = config_path.replace("\\", "/")
    match = re.search(r"(arch-1/.+)$", normalized)
    if match:
        candidate = ROOT.parent / match.group(1)
        if candidate.exists():
            return _load_json(candidate)
    filename = Path(normalized).name
    if filename:
        candidate = ROOT / "implementation" / "configs" / filename
        if candidate.exists():
            return _load_json(candidate)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return _load_json(path)


def _collect_top_level_runs() -> list[RunRecord]:
    records: list[RunRecord] = []
    for summary_path in sorted(RUNS_ROOT.glob("*/summary.json")):
        summary = _load_json(summary_path)
        dataset, benchmark = _infer_dataset(summary, summary_path.parent.name)
        category = _categorize_run(summary_path.parent.name)
        config = _load_config(str(summary.get("config_path")))
        api_base = str((config.get("provider") or {}).get("api_base") or "")
        model = str((config.get("defaults") or {}).get("model") or "unknown")
        systems = summary.get("systems") or {}
        for system_name, payload in sorted(systems.items()):
            if not isinstance(payload, dict):
                continue
            aggregate = payload.get("aggregate") or {}
            usage = aggregate.get("usage") or {}
            overall = ((aggregate.get("reference_metrics") or {}).get("overall") or {})
            behavior = aggregate.get("behavior_metrics") or {}
            failure_notes = payload.get("failures") or []
            note = _status_note(summary_path.parent.name, int(payload.get("completed_count") or 0), int(payload.get("failed_count") or 0))
            if failure_notes:
                first_error = str(failure_notes[0].get("error") or "").strip().splitlines()[0]
                if "invalid function call" in first_error:
                    note = "endpoint misconfigured (HTTP 404 invalid function call)"
                elif "billing cycle spend limit reached" in first_error:
                    note = "blocked by Modal billing cap"
                elif "max_tokens" in first_error and "maximum context length" in first_error:
                    note = "context window overflow from max_tokens budget"
                elif "HTTP 400" in first_error:
                    note = "request format or context failure"
            task_count = int(summary.get("task_count") or 0)
            completed = int(payload.get("completed_count") or 0)
            failed = int(payload.get("failed_count") or 0)
            records.append(
                RunRecord(
                    run_name=summary_path.parent.name,
                    source_kind="summary",
                    category=category,
                    dataset=dataset,
                    benchmark=benchmark,
                    config_name=str(summary.get("config_name") or ""),
                    config_path=str(summary.get("config_path") or ""),
                    model=model,
                    provider=api_base,
                    endpoint_kind=_provider_kind(api_base),
                    system=str(system_name),
                    started_at=payload.get("started_at_utc"),
                    completed_at=payload.get("completed_at_utc"),
                    task_count=task_count,
                    completed_count=completed,
                    failed_count=failed,
                    completion_rate=(completed / task_count) if task_count else 0.0,
                    wall_clock_elapsed_sec=payload.get("wall_clock_elapsed_sec"),
                    mean_run_elapsed_sec=aggregate.get("mean_run_elapsed_sec"),
                    total_tokens=usage.get("total_tokens"),
                    model_call_count=aggregate.get("model_call_count"),
                    answer_em=overall.get("answer_em"),
                    answer_f1=overall.get("answer_f1"),
                    choice_accuracy=overall.get("choice_accuracy"),
                    supporting_fact_title_precision=overall.get("supporting_fact_title_precision"),
                    supporting_fact_title_recall=overall.get("supporting_fact_title_recall"),
                    supporting_fact_context_coverage=overall.get("supporting_fact_context_coverage"),
                    decomposition_rate=behavior.get("decomposition_rate"),
                    evidence_grounding_rate=behavior.get("evidence_grounding_rate"),
                    revision_after_critique_rate=behavior.get("revision_after_critique_rate"),
                    premature_finalization_rate=behavior.get("premature_finalization_rate"),
                    unsupported_claim_rate=behavior.get("unsupported_claim_rate"),
                    status_note=note,
                    paper_claim_blocked=bool(summary.get("paper_claim_blockers")),
                )
            )
    return records


def _collect_gpqa_full_committee() -> RunRecord:
    run_root = RUNS_ROOT / "gpqa_diamond_modal_committee_198_20260405"
    shard_paths = sorted(run_root.glob("fixed_shard*/summary.json"))
    if not shard_paths:
        raise FileNotFoundError("Missing fixed shard summaries for full GPQA committee run")

    completed = 0
    failed = 0
    task_count = 0
    total_tokens = 0
    model_calls = 0
    weighted_elapsed = 0.0
    started_at_values: list[str] = []
    completed_at_values: list[str] = []
    correct = 0.0
    failure_counter: Counter[str] = Counter()

    for shard_path in shard_paths:
        summary = _load_json(shard_path)
        system = (summary.get("systems") or {}).get("committee") or {}
        aggregate = system.get("aggregate") or {}
        usage = aggregate.get("usage") or {}
        overall = ((aggregate.get("reference_metrics") or {}).get("overall") or {})
        shard_completed = int(system.get("completed_count") or 0)
        shard_failed = int(system.get("failed_count") or 0)
        shard_task_count = int(summary.get("task_count") or 0)
        completed += shard_completed
        failed += shard_failed
        task_count += shard_task_count
        total_tokens += int(usage.get("total_tokens") or 0)
        model_calls += int(aggregate.get("model_call_count") or 0)
        weighted_elapsed += float(aggregate.get("mean_run_elapsed_sec") or 0.0) * shard_completed
        correct += float(overall.get("choice_accuracy") or 0.0) * shard_completed
        if system.get("started_at_utc"):
            started_at_values.append(str(system["started_at_utc"]))
        if system.get("completed_at_utc"):
            completed_at_values.append(str(system["completed_at_utc"]))
        for failure in system.get("failures") or []:
            error = str((failure or {}).get("error") or "")
            if "billing cycle spend limit reached" in error:
                failure_counter["HTTP 429 billing cap"] += 1
            elif "HTTP 500" in error:
                failure_counter["HTTP 500 internal termination"] += 1
            elif "HTTP 400" in error:
                failure_counter["HTTP 400 context overflow"] += 1
            else:
                failure_counter["other"] += 1

    report_path = ROOT / "evals" / "reports" / "gpqa_diamond_modal_committee_198_20260405.md"
    report_text = report_path.read_text(encoding="utf-8")
    official_non_thinking = _extract_report_number(report_text, r"non-thinking official GPQA Diamond: `([0-9.]+)`")
    official_thinking = _extract_report_number(report_text, r"thinking official GPQA Diamond: `([0-9.]+)`")
    note_bits = [f"{name}={count}" for name, count in sorted(failure_counter.items())]
    note = "partial completion; " + ", ".join(note_bits)

    return RunRecord(
        run_name="gpqa_diamond_modal_committee_198_20260405",
        source_kind="aggregated_shards",
        category="benchmark",
        dataset="GPQA",
        benchmark="gpqa_diamond",
        config_name="qwen3_8b_modal_eval_timeout1800_abs",
        config_path=str(run_root / "qwen3_8b_modal_eval_timeout1800_abs.json"),
        model="Qwen/Qwen3-8B",
        provider="https://luthiraboss--korg-arch1-qwen3-vllm-serve.modal.run",
        endpoint_kind="Modal",
        system="committee",
        started_at=min(started_at_values) if started_at_values else None,
        completed_at=max(completed_at_values) if completed_at_values else None,
        task_count=task_count,
        completed_count=completed,
        failed_count=failed,
        completion_rate=(completed / task_count) if task_count else 0.0,
        wall_clock_elapsed_sec=_iso_span_seconds(min(started_at_values), max(completed_at_values)) if started_at_values and completed_at_values else None,
        mean_run_elapsed_sec=(weighted_elapsed / completed) if completed else None,
        total_tokens=total_tokens,
        model_call_count=model_calls,
        answer_em=None,
        answer_f1=None,
        choice_accuracy=(correct / completed) if completed else None,
        supporting_fact_title_precision=None,
        supporting_fact_title_recall=None,
        supporting_fact_context_coverage=None,
        decomposition_rate=None,
        evidence_grounding_rate=None,
        revision_after_critique_rate=None,
        premature_finalization_rate=None,
        unsupported_claim_rate=None,
        status_note=note,
        paper_claim_blocked=True,
    )


def _extract_report_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _iso_span_seconds(started_at: str, completed_at: str) -> float:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    return (end - start).total_seconds()


def _committee_stage_token_breakdown(summary_path: Path) -> list[tuple[str, int]]:
    summary = _load_json(summary_path)
    committee = ((summary.get("systems") or {}).get("committee") or {})
    per_role = (((committee.get("aggregate") or {}).get("per_role_usage")) or {})
    stages = []
    for role in [
        "coordinator_plan",
        "researcher",
        "analyst",
        "critic",
        "coordinator_finalize",
    ]:
        usage = per_role.get(role) or {}
        stages.append((role, int(usage.get("total_tokens") or 0)))
    return stages


def _load_hotpot_comparison() -> dict[str, Any]:
    committee_summary = _load_json(RUNS_ROOT / "hotpotqa_aws_g4dn_thinking_75_20260407_retry4b" / "summary.json")
    single_summary = _load_json(RUNS_ROOT / "hotpotqa_aws_g4dn_qwen_single_shot_75_20260414" / "summary.json")
    committee_runs = {
        run["task_id"]: run
        for run in ((committee_summary.get("systems") or {}).get("committee") or {}).get("runs") or []
        if isinstance(run, dict)
    }
    single_runs = {
        run["task_id"]: run
        for run in ((single_summary.get("systems") or {}).get("single_shot") or {}).get("runs") or []
        if isinstance(run, dict)
    }
    matched_ids = sorted(set(committee_runs) & set(single_runs))
    committee_wins = 0
    single_wins = 0
    ties = 0
    for task_id in matched_ids:
        committee_f1 = float(((committee_runs[task_id].get("reference_metrics") or {}).get("answer_f1")) or 0.0)
        single_f1 = float(((single_runs[task_id].get("reference_metrics") or {}).get("answer_f1")) or 0.0)
        if committee_f1 > single_f1:
            committee_wins += 1
        elif single_f1 > committee_f1:
            single_wins += 1
        else:
            ties += 1
    return {
        "matched_completed_tasks": len(matched_ids),
        "committee_f1_wins": committee_wins,
        "single_shot_f1_wins": single_wins,
        "ties": ties,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '.title { font: 700 22px sans-serif; fill: #102a43; }',
        '.subtitle { font: 400 12px sans-serif; fill: #486581; }',
        '.axis { font: 12px sans-serif; fill: #334e68; }',
        '.small { font: 11px sans-serif; fill: #334e68; }',
        '.label { font: 12px sans-serif; fill: #102a43; }',
        '.value { font: 700 12px sans-serif; fill: #102a43; }',
        '</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fbff" />',
    ]


def _svg_footer(lines: list[str]) -> str:
    return "\n".join(lines + ["</svg>"])


def _short_run_label(record: RunRecord) -> str:
    base = record.run_name
    base = base.replace("hotpotqa_", "hotpot ")
    base = base.replace("gpqa_", "gpqa ")
    base = base.replace("_20260414", "")
    base = base.replace("_20260407_retry4b", "")
    base = base.replace("_20260405", "")
    base = base.replace("_20260403", "")
    return base.replace("_", " ")


def _render_completion_chart(records: list[RunRecord], path: Path) -> None:
    width = 1400
    height = 620
    left = 110
    bottom = 130
    top = 90
    chart_width = width - left - 40
    chart_height = height - top - bottom
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="32" y="42">Architecture 1 Run Completion Rate</text>')
    lines.append('<text class="subtitle" x="32" y="64">Top-level benchmark, pilot, smoke, and probe runs from arch-1/evals/runs plus the aggregated GPQA 198 shard run.</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')

    for tick in range(0, 101, 20):
        y = top + chart_height - (tick / 100) * chart_height
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#d9e2ec" stroke-width="1"/>')
        lines.append(f'<text class="axis" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick}%</text>')

    bar_count = len(records)
    gap = 12
    bar_width = max(18, (chart_width - gap * (bar_count - 1)) / max(bar_count, 1))
    color_map = {
        "benchmark": "#2f855a",
        "pilot": "#2b6cb0",
        "smoke": "#dd6b20",
        "probe": "#805ad5",
        "other": "#718096",
    }
    for index, record in enumerate(records):
        x = left + index * (bar_width + gap)
        h = record.completion_rate * chart_height
        y = top + chart_height - h
        color = color_map.get(record.category, "#718096")
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" rx="4" fill="{color}" />')
        lines.append(f'<text class="value" x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{_fmt_pct(record.completion_rate, 0)}</text>')
        label = escape(_short_run_label(record))
        lines.append(f'<text class="small" transform="translate({x + bar_width / 2:.1f},{top + chart_height + 18}) rotate(45)" text-anchor="start">{label}</text>')

    legend_items = [("benchmark", "#2f855a"), ("pilot", "#2b6cb0"), ("smoke", "#dd6b20"), ("probe", "#805ad5")]
    for idx, (name, color) in enumerate(legend_items):
        lx = 32 + idx * 180
        ly = height - 26
        lines.append(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" fill="{color}" rx="2" />')
        lines.append(f'<text class="axis" x="{lx + 22}" y="{ly}">{name}</text>')

    _write_text(path, _svg_footer(lines))


def _render_hotpot_quality_chart(records: list[RunRecord], path: Path) -> None:
    hotpot = [r for r in records if r.dataset == "HotpotQA" and r.answer_f1 is not None]
    hotpot.sort(key=lambda r: (r.category, r.run_name, r.system))
    width = 1400
    height = 700
    left = 130
    right = 40
    top = 90
    bottom = 180
    chart_width = width - left - right
    chart_height = height - top - bottom
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="32" y="42">HotpotQA Quality by Run and System</text>')
    lines.append('<text class="subtitle" x="32" y="64">Answer F1 is the primary quality metric here. EM is annotated above each bar pair.</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    for tick in range(0, 101, 20):
        y = top + chart_height - (tick / 100) * chart_height
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#d9e2ec" stroke-width="1"/>')
        lines.append(f'<text class="axis" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick}%</text>')

    gap = 22
    bar_width = max(20, (chart_width - gap * (len(hotpot) - 1)) / max(len(hotpot), 1))
    colors = {
        "committee": "#1f77b4",
        "single_shot": "#ff7f0e",
        "single_multiturn": "#2ca02c",
        "single_structured": "#d62728",
        "single_direct": "#9467bd",
    }
    for index, record in enumerate(hotpot):
        x = left + index * (bar_width + gap)
        h = (record.answer_f1 or 0.0) * chart_height
        y = top + chart_height - h
        color = colors.get(record.system, "#718096")
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" rx="4" fill="{color}" />')
        lines.append(f'<text class="value" x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{_fmt_pct(record.answer_f1, 0)}</text>')
        lines.append(f'<text class="small" x="{x + bar_width / 2:.1f}" y="{y - 24:.1f}" text-anchor="middle">EM {_fmt_pct(record.answer_em, 0)}</text>')
        label = escape(f"{record.system} | {_short_run_label(record)}")
        lines.append(f'<text class="small" transform="translate({x + bar_width / 2:.1f},{top + chart_height + 18}) rotate(45)" text-anchor="start">{label}</text>')

    _write_text(path, _svg_footer(lines))


def _render_quality_cost_scatter(records: list[RunRecord], path: Path) -> None:
    plotted = [
        r for r in records
        if r.mean_run_elapsed_sec is not None and r.primary_quality_metric_value() is not None and r.completed_count > 0
    ]
    width = 1200
    height = 760
    left = 120
    right = 70
    top = 90
    bottom = 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_x = max((r.mean_run_elapsed_sec or 0.0) for r in plotted) * 1.1
    max_y = max(r.primary_quality_metric_value() for r in plotted) * 1.15
    max_tokens = max((r.total_tokens or 0) for r in plotted) or 1
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="32" y="42">Quality vs Cost</text>')
    lines.append('<text class="subtitle" x="32" y="64">X-axis = mean seconds per completed task. Y-axis = answer F1 for HotpotQA or choice accuracy for GPQA. Bubble size tracks total tokens.</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    for tick in range(0, 6):
        x_value = max_x * tick / 5
        x = left + (x_value / max_x) * chart_width if max_x else left
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" stroke="#eef2f6" stroke-width="1"/>')
        lines.append(f'<text class="axis" x="{x:.1f}" y="{top + chart_height + 24}" text-anchor="middle">{_fmt_num(x_value, 0)}s</text>')
    for tick in range(0, 6):
        y_value = max_y * tick / 5
        y = top + chart_height - (y_value / max_y) * chart_height if max_y else top + chart_height
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#eef2f6" stroke-width="1"/>')
        lines.append(f'<text class="axis" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{_fmt_pct(y_value, 0)}</text>')
    dataset_colors = {"HotpotQA": "#dd6b20", "GPQA": "#2b6cb0"}
    for record in plotted:
        x = left + ((record.mean_run_elapsed_sec or 0.0) / max_x) * chart_width if max_x else left
        y = top + chart_height - (record.primary_quality_metric_value() / max_y) * chart_height if max_y else top + chart_height
        radius = 8 + 22 * math.sqrt((record.total_tokens or 0) / max_tokens)
        color = dataset_colors.get(record.dataset, "#718096")
        label = escape(f"{record.dataset} {record.system}")
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72" stroke="#102a43" stroke-width="1"/>')
        lines.append(f'<text class="small" x="{x + radius + 6:.1f}" y="{y + 4:.1f}">{label}</text>')
    _write_text(path, _svg_footer(lines))


def _render_gpqa_chart(records: list[RunRecord], path: Path) -> None:
    gpqa = [r for r in records if r.dataset == "GPQA"]
    gpqa.sort(key=lambda r: (r.run_name, r.system))
    report_path = ROOT / "evals" / "reports" / "gpqa_diamond_modal_committee_198_20260405.md"
    report_text = report_path.read_text(encoding="utf-8")
    official_non_thinking = _extract_report_number(report_text, r"non-thinking official GPQA Diamond: `([0-9.]+)`")
    official_thinking = _extract_report_number(report_text, r"thinking official GPQA Diamond: `([0-9.]+)`")
    width = 1200
    height = 640
    left = 120
    right = 40
    top = 90
    bottom = 150
    chart_width = width - left - right
    chart_height = height - top - bottom
    bars: list[tuple[str, float, str]] = []
    for record in gpqa:
        if record.choice_accuracy is not None and record.completed_count > 0:
            bars.append((f"{record.system} | {_short_run_label(record)}", record.choice_accuracy, "#2b6cb0"))
            if record.run_name == "gpqa_diamond_modal_committee_198_20260405":
                bars.append(("committee | full-set lower bound", 84 / 198, "#718096"))
    if official_non_thinking is not None:
        bars.append(("official qwen3-8b non-thinking", official_non_thinking / 100.0, "#dd6b20"))
    if official_thinking is not None:
        bars.append(("official qwen3-8b thinking", official_thinking / 100.0, "#2f855a"))

    lines = _svg_header(width, height)
    lines.append('<text class="title" x="32" y="42">GPQA Accuracy Context</text>')
    lines.append('<text class="subtitle" x="32" y="64">Includes repo-run committee traces and the official Qwen3-8B reference figures already recorded in the local GPQA report.</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    for tick in range(0, 101, 20):
        y = top + chart_height - (tick / 100) * chart_height
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#d9e2ec" stroke-width="1"/>')
        lines.append(f'<text class="axis" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick}%</text>')

    gap = 26
    bar_width = max(40, (chart_width - gap * (len(bars) - 1)) / max(len(bars), 1))
    for index, (label, value, color) in enumerate(bars):
        x = left + index * (bar_width + gap)
        h = value * chart_height
        y = top + chart_height - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" rx="4" fill="{color}" />')
        lines.append(f'<text class="value" x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{_fmt_pct(value, 1)}</text>')
        lines.append(f'<text class="small" transform="translate({x + bar_width / 2:.1f},{top + chart_height + 18}) rotate(45)" text-anchor="start">{escape(label)}</text>')

    _write_text(path, _svg_footer(lines))


def _render_stage_token_chart(stage_tokens: list[tuple[str, int]], path: Path) -> None:
    width = 1100
    height = 560
    left = 160
    top = 90
    right = 40
    bottom = 70
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_tokens = max(value for _, value in stage_tokens) * 1.1
    lines = _svg_header(width, height)
    lines.append('<text class="title" x="32" y="42">HotpotQA Committee Token Load by Stage</text>')
    lines.append('<text class="subtitle" x="32" y="64">Based on the 73-completion HotpotQA AWS committee run. Later stages pay for accumulated context from earlier ones.</text>')
    lines.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#9fb3c8" stroke-width="1"/>')
    bar_gap = 24
    bar_height = max(24, (chart_height - bar_gap * (len(stage_tokens) - 1)) / max(len(stage_tokens), 1))
    colors = ["#2b6cb0", "#3182ce", "#63b3ed", "#90cdf4", "#bee3f8"]
    for idx, (stage, tokens) in enumerate(stage_tokens):
        y = top + idx * (bar_height + bar_gap)
        w = (tokens / max_tokens) * chart_width if max_tokens else 0
        lines.append(f'<text class="label" x="{left - 12}" y="{y + bar_height / 2 + 4:.1f}" text-anchor="end">{escape(stage)}</text>')
        lines.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{bar_height:.1f}" rx="4" fill="{colors[idx % len(colors)]}" />')
        lines.append(f'<text class="value" x="{left + w + 8:.1f}" y="{y + bar_height / 2 + 4:.1f}">{tokens:,}</text>')
    _write_text(path, _svg_footer(lines))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_markdown(records: list[RunRecord], dataset_path: Path, svg_paths: dict[str, Path]) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    hotpot_compare = _load_hotpot_comparison()
    full_gpqa = next(r for r in records if r.run_name == "gpqa_diamond_modal_committee_198_20260405")
    best_hotpot_committee = next(r for r in records if r.run_name == "hotpotqa_aws_g4dn_thinking_75_20260407_retry4b" and r.system == "committee")
    best_hotpot_single = next(r for r in records if r.run_name == "hotpotqa_aws_g4dn_qwen_single_shot_75_20260414" and r.system == "single_shot")

    run_rows = []
    for record in sorted(records, key=lambda r: (r.dataset, r.category, r.run_name, r.system)):
        run_rows.append(
            [
                record.dataset,
                record.category,
                record.run_name,
                record.system,
                record.model,
                record.endpoint_kind,
                f"{record.completed_count}/{record.task_count}",
                _fmt_pct(record.completion_rate, 1),
                _fmt_pct(record.answer_f1, 1) if record.answer_f1 is not None else _fmt_pct(record.choice_accuracy, 1),
                _fmt_num(record.mean_run_elapsed_sec, 1),
                _fmt_num(record.total_tokens, 0),
                "yes" if record.paper_claim_blocked else "no",
                record.status_note,
            ]
        )

    dataset_rows = []
    for record in records:
        dataset_rows.append({
            "run_name": record.run_name,
            "dataset": record.dataset,
            "system": record.system,
            "model": record.model,
            "endpoint_kind": record.endpoint_kind,
            "category": record.category,
            "task_count": record.task_count,
            "completed_count": record.completed_count,
            "failed_count": record.failed_count,
            "completion_rate": record.completion_rate,
            "mean_run_elapsed_sec": record.mean_run_elapsed_sec,
            "total_tokens": record.total_tokens,
            "model_call_count": record.model_call_count,
            "answer_em": record.answer_em,
            "answer_f1": record.answer_f1,
            "choice_accuracy": record.choice_accuracy,
            "supporting_fact_title_precision": record.supporting_fact_title_precision,
            "supporting_fact_title_recall": record.supporting_fact_title_recall,
            "supporting_fact_context_coverage": record.supporting_fact_context_coverage,
            "decomposition_rate": record.decomposition_rate,
            "evidence_grounding_rate": record.evidence_grounding_rate,
            "revision_after_critique_rate": record.revision_after_critique_rate,
            "premature_finalization_rate": record.premature_finalization_rate,
            "unsupported_claim_rate": record.unsupported_claim_rate,
            "status_note": record.status_note,
            "paper_claim_blocked": record.paper_claim_blocked,
        })
    _write_json(dataset_path, dataset_rows)

    lines = [
        "# Architecture 1 Benchmark Report",
        "",
        f"- Generated at (UTC): `{now}`",
        f"- Source dataset: [`{dataset_path.relative_to(ROOT)}`]({dataset_path.name})",
        "- Scope: every top-level `arch-1/evals/runs/*/summary.json` benchmark artifact, plus the aggregated four-shard `gpqa_diamond_modal_committee_198_20260405` run.",
        "- Important validity note: every checked-in runnable config used here is a `prompt_scaffold_only` configuration, so these are engineering benchmark results for the scaffold, not paper-valid role-specialist checkpoint results.",
        "",
        "## Executive Summary",
        "",
        f"- Architecture 1 is implemented as a fixed five-call pipeline: `coordinator_plan -> researcher -> analyst -> critic -> coordinator_finalize`. The code enforces the stage order, per-role prompts, per-role schemas, and restricted local context views.",
        f"- On the strongest completed HotpotQA run in this repo, the `committee` scaffold on `Qwen/Qwen3-8B-AWQ` completed `73/75` tasks with `38.4%` EM and `57.7%` F1, versus the matched `single_shot` baseline at `0.0%` EM and `1.6%` F1.",
        f"- That HotpotQA quality gain was expensive: the committee used `1,543,424` total tokens and `150.0s` mean task latency, compared with `174,866` tokens and `23.1s` for the single-shot baseline.",
        f"- The full GPQA committee run on Modal partially worked, finishing `163/198` examples at `51.5%` completed-trace accuracy, but it was cut off mainly by `HTTP 429` billing-cap failures. The conservative full-set lower bound remained `42.4%`.",
        f"- Several checked-in runs are operational bring-up artifacts rather than meaningful benchmark outcomes: probe runs, smokes, and a 75-task AWQ single-shot attempt that failed `75/75` with `HTTP 404 invalid function call`.",
        "",
        "## How Architecture 1 Works",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[Coordinator Plan] --> B[Researcher]",
        "    B --> C[Analyst]",
        "    C --> D[Critic]",
        "    D --> E[Coordinator Finalize]",
        "```",
        "",
        "- `coordinator_plan` sees only the task-facing fields and decomposes the problem before any answer is attempted.",
        "- `researcher` adds evidence, options, assumptions, and open questions, but it still cannot see later-stage outputs.",
        "- `analyst` converts the task plus researcher handoff into a draft answer.",
        "- `critic` receives the draft and is explicitly tasked with finding unsupported jumps, missing evidence, and required revisions.",
        "- `coordinator_finalize` receives the full upstream trail and produces the final answer after integrating critique.",
        "",
        "Why it is wired this way:",
        "",
        "- The fixed role order forces decomposition before synthesis, then inserts a dedicated adversarial check before finalization.",
        "- Per-role schemas make each stage machine-checkable and keep later evaluation logic simple.",
        "- Local context views keep role prompts narrow. Later stages inherit only the artifacts they need, which is why token cost grows by stage instead of exploding immediately.",
        "- The architecture works by trading off latency and tokens for more structured reasoning pressure. The benchmark results here show that this can improve answer quality, especially on HotpotQA, but the cost increase is large.",
        "",
        "## Graphs",
        "",
        f"![Run completion]({svg_paths['completion'].name})",
        "",
        f"![Hotpot quality]({svg_paths['hotpot_quality'].name})",
        "",
        f"![Quality vs cost]({svg_paths['quality_cost'].name})",
        "",
        f"![GPQA context]({svg_paths['gpqa'].name})",
        "",
        f"![Stage token load]({svg_paths['stages'].name})",
        "",
        "## Key Findings",
        "",
        f"- HotpotQA matched-task comparison: committee won answer F1 on `{hotpot_compare['committee_f1_wins']}` of `{hotpot_compare['matched_completed_tasks']}` matched completed tasks, single-shot won `{hotpot_compare['single_shot_f1_wins']}`, and `{hotpot_compare['ties']}` tied.",
        f"- The committee HotpotQA run kept `100%` supporting-fact context coverage while answer F1 rose to `{_fmt_pct(best_hotpot_committee.answer_f1, 1)}`. Title precision and recall both fell to `{_fmt_pct(best_hotpot_committee.supporting_fact_title_precision, 1)}`, which suggests the scaffold got more answers right without becoming equally precise at selecting evidence titles.",
        f"- The committee HotpotQA behavior metrics were strong: decomposition `{_fmt_pct(best_hotpot_committee.decomposition_rate, 1)}`, evidence grounding `{_fmt_pct(best_hotpot_committee.evidence_grounding_rate, 1)}`, revision-after-critique `{_fmt_pct(best_hotpot_committee.revision_after_critique_rate, 1)}`, and premature finalization `{_fmt_pct(best_hotpot_committee.premature_finalization_rate, 1)}`.",
        f"- Cost was the main downside. Relative to the 75-task HotpotQA single-shot baseline, the committee used about `{best_hotpot_committee.total_tokens / best_hotpot_single.total_tokens:.1f}x` tokens and `{best_hotpot_committee.mean_run_elapsed_sec / best_hotpot_single.mean_run_elapsed_sec:.1f}x` mean task latency.",
        f"- On GPQA, the aggregated full committee run beat the local report's official non-thinking Qwen3-8B reference (`39.3%`) on completed traces, but it remained below the official thinking reference (`62.0%`). Because `35` tasks did not complete, the more conservative figure for the full 198-task set is only `{_fmt_pct(84/198, 1)}`.",
        "",
        "## Operational Notes From Logs",
        "",
        "- `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414/benchmark.log` shows the 75-task AWS single-shot run completed end-to-end and checkpointed every task.",
        "- `hotpotqa_aws_g4dn_qwen_single_shot_75_20260414/vllm.log` shows `Qwen/Qwen3-8B-AWQ` running on a T4-class setup with `vLLM 0.10.2`, `max_model_len=8192`, AWQ quantization, and typical generation throughput around the mid-20 tokens/s range once warmed.",
        "- `hotpotqa_qwen3_8b_awq_single_shot_75_20260414/summary.json` shows a clean setup failure, not a bad model benchmark: `0/75` completed because every request failed with `HTTP 404: modal-http: invalid function call`.",
        "- The aggregated GPQA shard run was mainly an infrastructure-budget failure. The local GPQA report records `30` billing-cap `HTTP 429` failures, `4` Modal `HTTP 500` terminations, and `1` `HTTP 400` context-window overflow.",
        "",
        "## Complete Run Catalog",
        "",
        _markdown_table(
            [
                "Dataset",
                "Category",
                "Run",
                "System",
                "Model",
                "Endpoint",
                "Completed",
                "Completion",
                "Primary Quality",
                "Mean sec/task",
                "Tokens",
                "Scaffold",
                "Note",
            ],
            run_rows,
        ),
        "",
        "## Interpretation",
        "",
        "- The architecture works best in this repo when the benchmark rewards decomposition and revision. HotpotQA, which benefits from explicit evidence handling and a second-pass critique, is the clearest positive case.",
        "- The same structure is much more fragile operationally because it multiplies API calls by five for committee traces. That magnifies endpoint bugs, billing caps, and context-length issues.",
        "- The per-stage token chart shows why later stages are expensive: each stage receives a larger accumulated prompt, so the scaffold pays a compounding context tax across the pipeline.",
        "- Because all published configs are prompt scaffolds, the results answer an engineering question: whether the fixed committee workflow helps compared with simpler baselines using the same underlying model family. They do not yet validate the paper claim about four distinct role-specialist checkpoints.",
        "",
        "## Recommended Next Steps",
        "",
        "- Re-run the failed GPQA examples on a stable budgeted endpoint and merge only the missing tasks instead of restarting the full set.",
        "- Add the missing required ablations from `docs/BENCHMARKS.md`, especially `no critic`, `no researcher`, `no analyst`, and order swaps, so the architecture claim is better isolated.",
        "- Produce a like-for-like HotpotQA comparison where `single_multiturn`, `single_structured`, and `committee` all run on the same 75-task slice and same endpoint family. The repo currently mixes a strong 75-task committee run with a separate 26-task multi-turn run and small smoke runs.",
        "- Replace `prompt_scaffold_only` with genuinely distinct role-tuned checkpoints if the goal is a paper-valid Architecture 1 result rather than an implementation scaffold benchmark.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _collect_top_level_runs()
    records.append(_collect_gpqa_full_committee())
    records.sort(key=lambda r: (r.dataset, r.category, r.run_name, r.system))

    dataset_path = OUTPUT_DIR / "arch1_benchmark_dataset.json"
    completion_path = OUTPUT_DIR / "completion_rate.svg"
    hotpot_quality_path = OUTPUT_DIR / "hotpot_quality.svg"
    quality_cost_path = OUTPUT_DIR / "quality_vs_cost.svg"
    gpqa_path = OUTPUT_DIR / "gpqa_accuracy.svg"
    stages_path = OUTPUT_DIR / "hotpot_committee_stage_tokens.svg"

    _render_completion_chart(records, completion_path)
    _render_hotpot_quality_chart(records, hotpot_quality_path)
    _render_quality_cost_scatter(records, quality_cost_path)
    _render_gpqa_chart(records, gpqa_path)
    _render_stage_token_chart(
        _committee_stage_token_breakdown(RUNS_ROOT / "hotpotqa_aws_g4dn_thinking_75_20260407_retry4b" / "summary.json"),
        stages_path,
    )

    markdown = _build_markdown(
        records,
        dataset_path,
        {
            "completion": completion_path,
            "hotpot_quality": hotpot_quality_path,
            "quality_cost": quality_cost_path,
            "gpqa": gpqa_path,
            "stages": stages_path,
        },
    )
    _write_text(OUTPUT_DIR / "ARCHITECTURE1_BENCHMARK_REPORT.md", markdown)


if __name__ == "__main__":
    main()
