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

from committee_llm.benchmark import main


class DummyClient:
    def __init__(self, **_: object):
        pass


class FakeCommitteeRunner:
    def __init__(self, config: object, client: object):
        self.config = config
        self.client = client

    def run_task(self, task: object) -> dict[str, object]:
        task_id = getattr(task, "task_id", None) or "task"
        benchmark_name = getattr(task, "benchmark_name", None) or "hotpotqa_distractor"
        return {
            "run_id": task_id,
            "architecture": "architecture_1_posttrained_committee",
            "config_name": "fake-config",
            "config_path": "fake-config-path",
            "started_at_utc": "2026-03-30T00:00:00+00:00",
            "total_elapsed_sec": 2.0,
            "input": getattr(task, "prompt", ""),
            "task": {
                "id": task_id,
                "benchmark_name": benchmark_name,
                "prompt": getattr(task, "prompt", ""),
            },
            "metadata": {},
            "steps": [],
            "final_stage": "Answer",
            "final_answer": "Answer",
            "evaluation": {
                "reference": {
                    "benchmark_name": benchmark_name,
                    "metrics": {"answer_em": 1.0, "answer_f1": 1.0},
                }
            },
            "summary": {
                "stage_count": 5,
                "model_call_count": 5,
                "per_role_usage": {"coordinator_plan": {"total_tokens": 100}},
                "usage": {"total_tokens": 100},
                "behavior_metrics": {"decomposition_rate": 1.0},
            },
        }


class FakeSingleShotRunner:
    def __init__(self, config: object, client: object, temperature: float | None = None, max_tokens: int | None = None):
        self.config = config
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run_task(self, task: object) -> dict[str, object]:
        task_id = getattr(task, "task_id", None) or "task"
        benchmark_name = getattr(task, "benchmark_name", None) or "hotpotqa_distractor"
        return {
            "run_id": task_id,
            "architecture": "single_shot_baseline",
            "config_name": "fake-config",
            "config_path": "fake-config-path",
            "started_at_utc": "2026-03-30T00:00:00+00:00",
            "total_elapsed_sec": 1.0,
            "input": getattr(task, "prompt", ""),
            "task": {
                "id": task_id,
                "benchmark_name": benchmark_name,
                "prompt": getattr(task, "prompt", ""),
            },
            "metadata": {},
            "steps": [],
            "final_stage": "Answer",
            "final_answer": "Answer",
            "evaluation": {
                "reference": {
                    "benchmark_name": benchmark_name,
                    "metrics": {"answer_em": 0.0, "answer_f1": 0.5},
                }
            },
            "summary": {
                "stage_count": 1,
                "model_call_count": 1,
                "per_role_usage": {"single_shot": {"total_tokens": 40}},
                "usage": {"total_tokens": 40},
            },
        }


class FakeSingleMultiTurnRunner(FakeSingleShotRunner):
    def run_task(self, task: object) -> dict[str, object]:
        payload = super().run_task(task)
        payload["architecture"] = "single_model_multiturn_baseline"
        payload["total_elapsed_sec"] = 1.5
        payload["summary"]["model_call_count"] = 3
        payload["summary"]["per_role_usage"] = {
            "single_multiturn_plan": {"total_tokens": 30},
            "single_multiturn_evidence": {"total_tokens": 30},
            "single_multiturn_answer": {"total_tokens": 30},
        }
        payload["summary"]["usage"] = {"total_tokens": 90}
        payload["evaluation"]["reference"]["metrics"] = {"answer_em": 0.5, "answer_f1": 0.75}
        return payload


class FakeSingleStructuredRunner(FakeSingleShotRunner):
    def run_task(self, task: object) -> dict[str, object]:
        payload = super().run_task(task)
        payload["architecture"] = "single_model_structured_baseline"
        payload["total_elapsed_sec"] = 1.2
        payload["summary"]["per_role_usage"] = {"single_structured": {"total_tokens": 60}}
        payload["summary"]["usage"] = {"total_tokens": 60}
        payload["evaluation"]["reference"]["metrics"] = {"answer_em": 0.25, "answer_f1": 0.5}
        return payload


class BenchmarkRunnerTests(unittest.TestCase):
    def test_benchmark_summary_compares_all_systems(self) -> None:
        temp_dir = ROOT / f".test-benchmark-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            tasks_path = temp_dir / "tasks.jsonl"
            tasks_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "task-1",
                                "benchmark_name": "hotpotqa_distractor",
                                "task_type": "open_qa",
                                "prompt": "Question 1?",
                                "reference_answers": ["Answer 1"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "task-2",
                                "benchmark_name": "hotpotqa_distractor",
                                "task_type": "open_qa",
                                "prompt": "Question 2?",
                                "reference_answers": ["Answer 2"],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            outdir = temp_dir / "runs"
            with patch("committee_llm.benchmark.OpenAICompatibleChatClient", DummyClient), patch(
                "committee_llm.benchmark.CommitteeRunner",
                FakeCommitteeRunner,
            ), patch(
                "committee_llm.benchmark.SingleShotRunner",
                FakeSingleShotRunner,
            ), patch(
                "committee_llm.benchmark.SingleModelMultiTurnRunner",
                FakeSingleMultiTurnRunner,
            ), patch(
                "committee_llm.benchmark.SingleModelStructuredRunner",
                FakeSingleStructuredRunner,
            ):
                exit_code = main(
                    [
                        "--config",
                        str(ROOT / "implementation" / "configs" / "qwen3_8b_committee.json"),
                        "--tasks",
                        str(tasks_path),
                        "--outdir",
                        str(outdir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["systems"]["committee"]["completed_count"], 2)
            self.assertEqual(summary["systems"]["single_shot"]["completed_count"], 2)
            self.assertEqual(summary["systems"]["single_multiturn"]["completed_count"], 2)
            self.assertEqual(summary["systems"]["single_structured"]["completed_count"], 2)
            self.assertEqual(
                summary["systems"]["committee"]["aggregate"]["reference_metrics"]["overall"]["answer_em"],
                1.0,
            )
            self.assertEqual(
                summary["systems"]["single_multiturn"]["aggregate"]["reference_metrics"]["overall"]["answer_f1"],
                0.75,
            )
            self.assertEqual(
                summary["systems"]["single_shot"]["aggregate"]["reference_metrics"]["overall"]["answer_f1"],
                0.5,
            )
            self.assertEqual(
                summary["comparison"]["committee_vs_single_shot"]["metric_deltas"]["answer_em"],
                1.0,
            )
            self.assertEqual(
                summary["comparison"]["committee_vs_single_shot"]["token_ratio"],
                2.5,
            )
            self.assertEqual(
                summary["systems"]["committee"]["aggregate"]["behavior_metrics"]["decomposition_rate"],
                1.0,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
