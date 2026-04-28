from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "implementation"))

from committee_llm.benchmark_data import convert_gpqa, convert_hotpotqa


class BenchmarkDataConversionTests(unittest.TestCase):
    def test_convert_hotpotqa_preserves_context_and_gold_answer(self) -> None:
        temp_dir = ROOT / f".test-benchmark-data-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            input_path = temp_dir / "hotpot.json"
            output_path = temp_dir / "hotpot.jsonl"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "_id": "hp-1",
                            "question": "Who wrote Hamlet?",
                            "answer": "William Shakespeare",
                            "type": "bridge",
                            "supporting_facts": [["Hamlet", 0]],
                            "context": [["Hamlet", ["Hamlet is a tragedy.", "It was written by William Shakespeare."]]],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            count = convert_hotpotqa(input_path, output_path)
            converted = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(count, 1)
            self.assertEqual(converted["benchmark_name"], "hotpotqa_distractor")
            self.assertEqual(converted["reference_answers"], ["William Shakespeare"])
            self.assertIn("William Shakespeare", converted["context"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_convert_gpqa_creates_shuffled_choices_and_correct_label(self) -> None:
        temp_dir = ROOT / f".test-benchmark-data-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            input_path = temp_dir / "gpqa.csv"
            output_path = temp_dir / "gpqa.jsonl"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Question ID", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "High-level domain"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question ID": "gpqa-1",
                        "Question": "Which option is correct?",
                        "Correct Answer": "Gamma",
                        "Incorrect Answer 1": "Alpha",
                        "Incorrect Answer 2": "Beta",
                        "Incorrect Answer 3": "Delta",
                        "High-level domain": "science",
                    }
                )

            count = convert_gpqa(input_path, output_path)
            converted = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(count, 1)
            self.assertEqual(converted["benchmark_name"], "gpqa")
            self.assertEqual(converted["task_type"], "multiple_choice")
            self.assertEqual(len(converted["choices"]), 4)
            self.assertIn(converted["correct_choice"], {"A", "B", "C", "D"})
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
