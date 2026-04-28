from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "implementation"))

from committee_llm.benchmark_report import main


class BenchmarkReportTests(unittest.TestCase):
    def test_render_markdown_report_contains_tables_and_failures(self) -> None:
        temp_dir = ROOT / f".test-benchmark-report-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            summary_path = temp_dir / "summary.json"
            output_path = temp_dir / "summary.md"
            summary_path.write_text(
                json.dumps(
                    {
                        "config_name": "qwen3_8b_committee_eval",
                        "config_path": str(ROOT / "implementation" / "configs" / "qwen3_8b_committee_eval.json"),
                        "task_count": 2,
                        "paper_claim_blockers": [
                            "Role specialization source is not set to post_trained_role_specialists, so this config should be treated as a prompt scaffold."
                        ],
                        "systems": {
                            "committee": {
                                "completed_count": 2,
                                "failed_count": 1,
                                "wall_clock_elapsed_sec": 5.0,
                                "aggregate": {
                                    "mean_run_elapsed_sec": 2.5,
                                    "tasks_per_sec": 0.4,
                                    "model_call_count": 10,
                                    "usage": {"total_tokens": 200},
                                    "behavior_metrics": {"decomposition_rate": 1.0},
                                    "reference_metrics": {
                                        "overall": {"answer_em": 0.5, "answer_f1": 0.75},
                                        "by_benchmark": {
                                            "hotpotqa_distractor": {
                                                "count": 2,
                                                "metrics": {"answer_em": 0.5, "answer_f1": 0.75},
                                            }
                                        },
                                    },
                                },
                                "failures": [{"task_id": "task-3", "error": "backend timeout"}],
                            },
                            "single_shot": {
                                "completed_count": 2,
                                "failed_count": 0,
                                "wall_clock_elapsed_sec": 3.0,
                                "aggregate": {
                                    "mean_run_elapsed_sec": 1.5,
                                    "tasks_per_sec": 0.6666667,
                                    "model_call_count": 2,
                                    "usage": {"total_tokens": 80},
                                    "behavior_metrics": {},
                                    "reference_metrics": {
                                        "overall": {"answer_em": 0.0, "answer_f1": 0.25},
                                        "by_benchmark": {
                                            "hotpotqa_distractor": {
                                                "count": 2,
                                                "metrics": {"answer_em": 0.0, "answer_f1": 0.25},
                                            }
                                        },
                                    },
                                },
                                "failures": [],
                            },
                        },
                        "comparison": {
                            "committee_vs_single_shot": {
                                "latency_ratio": 1.6666667,
                                "token_ratio": 2.5,
                                "metric_deltas": {"answer_em": 0.5, "answer_f1": 0.5},
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            exit_code = main(["--summary", str(summary_path), "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            report = output_path.read_text(encoding="utf-8")
            self.assertIn("# Benchmark Results", report)
            self.assertIn("Paper-claim blockers:", report)
            self.assertIn("| System | Completed | Failed | Wall Clock (s) | Mean Run (s) | Tasks/s | Total Tokens | Model Calls |", report)
            self.assertIn("| committee | 2 | 1 | 5 | 2.5 | 0.4 | 200 | 10 |", report)
            self.assertIn("| single_shot | 2 | 0 | 3 | 1.5 | 0.6667 | 80 | 2 |", report)
            self.assertIn("| System | answer_em | answer_f1 |", report)
            self.assertIn("| System | Benchmark | Count | answer_em | answer_f1 |", report)
            self.assertIn("| System | decomposition_rate |", report)
            self.assertIn("| Comparison | Latency Ratio | Token Ratio | answer_em Delta | answer_f1 Delta |", report)
            self.assertIn("| committee | task-3 | backend timeout |", report)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
