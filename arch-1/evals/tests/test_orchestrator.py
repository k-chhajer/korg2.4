from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "implementation"))

from committee_llm.client import ChatResult
from committee_llm.config import ExperimentConfig
from committee_llm.orchestrator import CommitteeRunner
from committee_llm.tasks import TaskSpec


class FakeClient:
    def __init__(self, responses: list[ChatResult]):
        self._responses = list(responses)

    def chat(self, **_: object) -> ChatResult:
        if not self._responses:
            raise AssertionError("No fake responses remaining.")
        return self._responses.pop(0)


class CommitteeRunnerSummaryTests(unittest.TestCase):
    def test_run_includes_aggregated_usage_summary(self) -> None:
        config = ExperimentConfig.load(ROOT / "implementation" / "configs" / "qwen3_8b_committee.json")
        runner = CommitteeRunner(
            config=config,
            client=FakeClient(
                [
                    ChatResult(
                        model="test-model",
                        content=(
                            "Problem Restatement:\nTask\nSuccess Criteria:\nCorrect\nPlan:\nStep\n"
                            "Risks and Unknowns:\nRisk\nResearch Brief:\nBrief"
                        ),
                        elapsed_sec=0.5,
                        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content=(
                            "Evidence and Observations:\nFacts\nOptions:\nOption A\nAssumptions:\nSome\n"
                            "Open Questions:\nNone\nResearch Handoff:\nReady"
                        ),
                        elapsed_sec=0.7,
                        usage={
                            "prompt_tokens": 11,
                            "completion_tokens": 21,
                            "total_tokens": 32,
                            "completion_tokens_details": {"reasoning_tokens": 5},
                        },
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content=(
                            "Core Answer:\nDraft\nReasoning:\nBecause\nCaveats:\nCaveat\n"
                            "Draft Deliverable:\nDeliverable"
                        ),
                        elapsed_sec=0.9,
                        usage={"prompt_tokens": 12, "completion_tokens": 22, "total_tokens": 34},
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content=(
                            "Major Issues:\nOne\nMinor Issues:\nTwo\nMissing Evidence:\nThree\n"
                            "Required Revisions:\nFour"
                        ),
                        elapsed_sec=1.1,
                        usage={
                            "prompt_tokens": 13,
                            "completion_tokens": 23,
                            "total_tokens": 36,
                            "completion_tokens_details": {"reasoning_tokens": 7},
                        },
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content=(
                            "Final Answer:\nFinal output\nDecision Rationale:\nGood enough\n"
                            "Residual Risks:\nLow"
                        ),
                        elapsed_sec=1.3,
                        usage={"prompt_tokens": 14, "completion_tokens": 24, "total_tokens": 38},
                        raw_response={},
                    ),
                ]
            ),
        )

        trace = runner.run("Test prompt", task_id="task-123")

        self.assertEqual(trace["run_id"], "task-123")
        self.assertEqual(trace["final_answer"], "Final output")
        self.assertEqual(trace["summary"]["stage_count"], 5)
        self.assertEqual(trace["summary"]["usage"]["prompt_tokens"], 60)
        self.assertEqual(trace["summary"]["usage"]["completion_tokens"], 110)
        self.assertEqual(trace["summary"]["usage"]["total_tokens"], 170)
        self.assertEqual(
            trace["summary"]["usage"]["completion_tokens_details"]["reasoning_tokens"],
            12,
        )
        self.assertAlmostEqual(trace["summary"]["per_role_elapsed_sec"]["critic"], 1.1)

    def test_run_task_includes_structured_task_and_evaluation(self) -> None:
        config = ExperimentConfig.load(ROOT / "implementation" / "configs" / "qwen3_8b_committee.json")
        runner = CommitteeRunner(
            config=config,
            client=FakeClient(
                [
                    ChatResult(
                        model="test-model",
                        content="Problem Restatement:\nTask\nSuccess Criteria:\nCorrect\nPlan:\nStep\nRisks and Unknowns:\nRisk\nResearch Brief:\nBrief",
                        elapsed_sec=0.1,
                        usage={},
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content="Evidence and Observations:\nFacts\nOptions:\nOption A\nAssumptions:\nSome\nOpen Questions:\nNone\nResearch Handoff:\nReady",
                        elapsed_sec=0.1,
                        usage={},
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content="Core Answer:\nDraft\nReasoning:\nBecause\nCaveats:\nCaveat\nDraft Deliverable:\nDeliverable",
                        elapsed_sec=0.1,
                        usage={},
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content="Major Issues:\nOne\nMinor Issues:\nTwo\nMissing Evidence:\nThree\nRequired Revisions:\nFour",
                        elapsed_sec=0.1,
                        usage={},
                        raw_response={},
                    ),
                    ChatResult(
                        model="test-model",
                        content="Final Answer:\nUse a baseline, match compute, and report a metric.\nDecision Rationale:\nGood enough\nResidual Risks:\nLow",
                        elapsed_sec=0.1,
                        usage={},
                        raw_response={},
                    ),
                ]
            ),
        )

        task = TaskSpec.from_dict(
            {
                "id": "research-123",
                "domain": "llm_eval",
                "prompt": "Design the experiment.",
                "context": "Paper methods section.",
                "objectives": ["Match compute"],
                "constraints": ["Keep it small"],
                "deliverable": "A short design note.",
                "evaluation_criteria": ["Includes baseline and metric"],
                "checks": {"must_contain": ["baseline", "compute", "metric"]},
                "metadata": {"split": "dev"},
            }
        )

        trace = runner.run_task(task)

        self.assertEqual(trace["task"]["domain"], "llm_eval")
        self.assertEqual(trace["metadata"]["split"], "dev")
        self.assertTrue(trace["evaluation"]["passed"])
        self.assertEqual(trace["evaluation"]["check_count"], 3)
        first_prompt = str(trace["steps"][0]["user_prompt"])
        self.assertIn("<task_context>", first_prompt)
        self.assertIn("<evaluation_criteria>", first_prompt)


if __name__ == "__main__":
    unittest.main()
