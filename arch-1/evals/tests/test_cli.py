from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "implementation"))

from committee_llm.cli import main


class DummyClient:
    def __init__(self, **_: object):
        pass


class FakeRunner:
    def __init__(self, config: object, client: object):
        self.config = config
        self.client = client

    def run_task(self, task: object) -> dict[str, object]:
        task_id = getattr(task, "task_id", None) or "run"
        domain = getattr(task, "domain", None) or "unspecified"
        suffix = 1 if task_id == "multi_hop_001" else 2
        return {
            "run_id": task_id,
            "architecture": "committee_v1_explicit_coordination",
            "config_name": "fake-config",
            "config_path": "fake-config-path",
            "started_at_utc": "2026-03-30T00:00:00+00:00",
            "total_elapsed_sec": 1.0 * suffix,
            "input": getattr(task, "prompt", ""),
            "task": {
                "id": task_id,
                "domain": domain,
                "prompt": getattr(task, "prompt", ""),
                "context": getattr(task, "context", None),
                "objectives": list(getattr(task, "objectives", [])),
                "constraints": list(getattr(task, "constraints", [])),
                "deliverable": getattr(task, "deliverable", None),
                "evaluation_criteria": list(getattr(task, "evaluation_criteria", [])),
                "references": list(getattr(task, "references", [])),
                "checks": dict(getattr(task, "checks", {})),
                "metadata": dict(getattr(task, "metadata", {})),
            },
            "metadata": dict(getattr(task, "metadata", {})),
            "steps": [],
            "final_stage": "Final Answer:\nAnswer\nDecision Rationale:\nBecause\nResidual Risks:\nLow",
            "final_answer": "Answer",
            "evaluation": {
                "passed": suffix == 1,
                "score": 1.0 if suffix == 1 else 0.5,
            },
            "summary": {
                "stage_count": 5,
                "per_role_elapsed_sec": {
                    "coordinator_plan": 0.1 * suffix,
                    "researcher": 0.2 * suffix,
                },
                "per_role_usage": {
                    "coordinator_plan": {
                        "prompt_tokens": 10 * suffix,
                        "completion_tokens": 20 * suffix,
                        "total_tokens": 30 * suffix,
                    },
                    "researcher": {
                        "prompt_tokens": 5 * suffix,
                        "completion_tokens": 15 * suffix,
                        "total_tokens": 20 * suffix,
                    },
                },
                "usage": {
                    "prompt_tokens": 15 * suffix,
                    "completion_tokens": 35 * suffix,
                    "total_tokens": 50 * suffix,
                },
            },
        }


class FlakyRunner(FakeRunner):
    def run_task(self, task: object) -> dict[str, object]:
        task_id = getattr(task, "task_id", None)
        if task_id == "architecture_001":
            raise RuntimeError("backend unavailable")
        return super().run_task(task)


class CliBatchSummaryTests(unittest.TestCase):
    def test_batch_mode_writes_summary_json(self) -> None:
        temp_dir = ROOT / f".test-cli-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            outdir = temp_dir / "runs"
            with patch("committee_llm.cli.OpenAICompatibleChatClient", DummyClient), patch(
                "committee_llm.cli.CommitteeRunner",
                FakeRunner,
            ):
                exit_code = main(
                    [
                        "--config",
                        str(ROOT / "implementation" / "configs" / "qwen3_8b_committee.json"),
                        "--tasks",
                        str(ROOT / "evals" / "data" / "sample_tasks.jsonl"),
                        "--outdir",
                        str(outdir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_count"], 2)
            self.assertEqual(summary["failed_count"], 0)
            self.assertEqual(summary["aggregate"]["usage"]["prompt_tokens"], 45)
            self.assertEqual(summary["aggregate"]["usage"]["total_tokens"], 150)
            self.assertEqual(summary["aggregate"]["per_role_usage"]["coordinator_plan"]["total_tokens"], 90)
            self.assertEqual(summary["aggregate"]["evaluation"]["passed_count"], 1)
            self.assertAlmostEqual(float(summary["aggregate"]["evaluation"]["mean_score"]), 0.75)
            self.assertEqual(
                summary["aggregate"]["evaluation"]["by_domain"]["llm_architecture_eval"]["completed_count"],
                2,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_continue_on_error_records_failures_and_keeps_running(self) -> None:
        temp_dir = ROOT / f".test-cli-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            outdir = temp_dir / "runs"
            with patch("committee_llm.cli.OpenAICompatibleChatClient", DummyClient), patch(
                "committee_llm.cli.CommitteeRunner",
                FlakyRunner,
            ):
                exit_code = main(
                    [
                        "--config",
                        str(ROOT / "implementation" / "configs" / "qwen3_8b_committee.json"),
                        "--tasks",
                        str(ROOT / "evals" / "data" / "sample_tasks.jsonl"),
                        "--outdir",
                        str(outdir),
                        "--continue-on-error",
                    ]
                )

            self.assertEqual(exit_code, 1)
            summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["status"], "partial_failure")
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["failures"][0]["task_id"], "architecture_001")
            self.assertTrue((outdir / "multi_hop_001.json").exists())
            self.assertFalse((outdir / "architecture_001.json").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
