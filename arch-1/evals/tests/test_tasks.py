from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "implementation"))

from committee_llm.evaluation import evaluate_answer
from committee_llm.tasks import TaskSpec


class TaskSpecTests(unittest.TestCase):
    def test_task_spec_parses_structured_research_fields(self) -> None:
        task = TaskSpec.from_dict(
            {
                "id": "research-001",
                "domain": "llm_eval",
                "benchmark_name": "hotpotqa_distractor",
                "benchmark_split": "dev",
                "task_type": "open_qa",
                "prompt": "Design an evaluation.",
                "context": "Paper setting.",
                "objectives": ["Control compute", "Define baselines"],
                "constraints": ["Keep it small"],
                "deliverable": "Methods note",
                "evaluation_criteria": ["Includes a baseline"],
                "references": ["HotpotQA"],
                "reference_answers": ["Baseline experiment"],
                "checks": {"must_contain": ["baseline"], "min_words": 50},
                "metadata": {"split": "dev"},
            }
        )

        self.assertEqual(task.task_id, "research-001")
        self.assertEqual(task.domain, "llm_eval")
        self.assertEqual(task.benchmark_name, "hotpotqa_distractor")
        self.assertEqual(task.task_type, "open_qa")
        self.assertEqual(task.objectives, ["Control compute", "Define baselines"])
        self.assertEqual(task.reference_answers, ["Baseline experiment"])
        self.assertEqual(task.checks["must_contain"], ["baseline"])
        self.assertEqual(task.metadata["split"], "dev")

    def test_evaluation_reports_deterministic_check_failures(self) -> None:
        task = TaskSpec.from_dict(
            {
                "id": "research-002",
                "prompt": "Write a design note.",
                "checks": {
                    "must_contain": ["baseline", "metric"],
                    "must_not_contain": ["magic"],
                    "min_words": 5,
                },
            }
        )

        evaluation = evaluate_answer(task, "This answer mentions a baseline but no scoring concept.")

        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["check_count"], 4)
        self.assertLess(float(evaluation["score"]), 1.0)

    def test_evaluation_computes_reference_metrics_for_open_qa_and_mcq(self) -> None:
        qa_task = TaskSpec.from_dict(
            {
                "id": "hotpot-1",
                "benchmark_name": "hotpotqa_distractor",
                "benchmark_split": "dev",
                "task_type": "open_qa",
                "prompt": "Who wrote Hamlet?",
                "context": "[1] Hamlet\nHamlet was written by William Shakespeare.",
                "reference_answers": ["William Shakespeare", "Shakespeare"],
                "metadata": {"supporting_facts": [["Hamlet", 0]]},
            }
        )
        qa_eval = evaluate_answer(qa_task, "William Shakespeare")
        self.assertEqual(qa_eval["reference"]["metrics"]["answer_em"], 1.0)
        self.assertEqual(qa_eval["reference"]["metrics"]["answer_f1"], 1.0)
        self.assertEqual(qa_eval["reference"]["metrics"]["supporting_fact_context_coverage"], 1.0)

        mcq_task = TaskSpec.from_dict(
            {
                "id": "gpqa-1",
                "benchmark_name": "gpqa",
                "task_type": "multiple_choice",
                "prompt": "Pick one.",
                "choices": [
                    {"label": "A", "text": "Alpha"},
                    {"label": "B", "text": "Beta"},
                    {"label": "C", "text": "Gamma"},
                    {"label": "D", "text": "Delta"},
                ],
                "correct_choice": "C",
            }
        )
        mcq_eval = evaluate_answer(mcq_task, "Answer: C. Gamma is correct.")
        self.assertEqual(mcq_eval["reference"]["predicted_choice"], "C")
        self.assertEqual(mcq_eval["reference"]["metrics"]["choice_accuracy"], 1.0)

    def test_evaluation_scores_leading_short_answer_phrase_in_explanatory_open_qa_output(self) -> None:
        yes_no_task = TaskSpec.from_dict(
            {
                "id": "hotpot-yes-no",
                "benchmark_name": "hotpotqa_distractor",
                "task_type": "open_qa",
                "prompt": "Were they the same nationality?",
                "reference_answers": ["yes"],
            }
        )
        yes_no_eval = evaluate_answer(
            yes_no_task,
            "Yes, both Scott Derrickson and Ed Wood were American.",
        )
        self.assertEqual(yes_no_eval["reference"]["metrics"]["answer_em"], 1.0)
        self.assertEqual(yes_no_eval["reference"]["scored_answer"], "Yes")

        entity_task = TaskSpec.from_dict(
            {
                "id": "hotpot-entity",
                "benchmark_name": "hotpotqa_distractor",
                "task_type": "open_qa",
                "prompt": "What position did she hold?",
                "reference_answers": ["Chief of Protocol"],
            }
        )
        entity_eval = evaluate_answer(
            entity_task,
            "Chief of Protocol. Shirley Temple later served in multiple diplomatic posts.",
        )
        self.assertEqual(entity_eval["reference"]["metrics"]["answer_em"], 1.0)
        self.assertEqual(entity_eval["reference"]["scored_answer"], "Chief of Protocol")

        prefix_task = TaskSpec.from_dict(
            {
                "id": "hotpot-prefix",
                "benchmark_name": "hotpotqa_distractor",
                "task_type": "open_qa",
                "prompt": "Which university?",
                "reference_answers": ["Oxford"],
            }
        )
        prefix_eval = evaluate_answer(
            prefix_task,
            "Oxford University. Kate Millett attended St Hilda's College there.",
        )
        self.assertEqual(prefix_eval["reference"]["metrics"]["answer_em"], 1.0)
        self.assertEqual(prefix_eval["reference"]["scored_answer"], "Oxford")


if __name__ == "__main__":
    unittest.main()
