from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _format_context_paragraphs(paragraphs: list[tuple[str, str]]) -> str:
    formatted = []
    for index, (title, text) in enumerate(paragraphs, start=1):
        clean_text = " ".join(text.split())
        formatted.append(f"[{index}] {title}\n{clean_text}")
    return "\n\n".join(formatted)


def _json_load_maybe(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def convert(input_path: str | Path, output_path: str | Path, *, limit: int | None = None) -> int:
    df = pd.read_parquet(input_path)
    rows = df.iloc[:limit] if limit is not None else df

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for _, row in rows.iterrows():
            context_raw = _json_load_maybe(row["context"])
            supporting_facts_raw = _json_load_maybe(row["supporting_facts"])
            evidences_raw = _json_load_maybe(row["evidences"])

            paragraphs: list[tuple[str, str]] = []
            titles: list[str] = []
            for item in context_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                title, sentences = item
                if not isinstance(title, str) or not isinstance(sentences, list):
                    continue
                text = " ".join(sentence.strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip())
                paragraphs.append((title, text))
                titles.append(title)

            supporting_facts: list[list[Any]] = []
            for item in supporting_facts_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                title, sent_id = item
                if isinstance(title, str):
                    supporting_facts.append([title, int(sent_id)])

            payload = {
                "id": str(row["_id"]),
                "benchmark_name": "2wikimultihopqa",
                "benchmark_split": "dev",
                "task_type": "open_qa",
                "domain": "multi_hop_qa",
                "prompt": str(row["question"]).strip(),
                "context": _format_context_paragraphs(paragraphs),
                "deliverable": (
                    "Answer the question using only the provided context. "
                    "Start with a short answer phrase, then briefly justify it."
                ),
                "evaluation_criteria": [
                    "Answer should match the gold answer under official normalization.",
                    "Reasoning should rely on the provided context.",
                ],
                "references": titles,
                "reference_answers": [str(row["answer"]).strip()],
                "metadata": {
                    "source_dataset": "2WikiMultihopQA",
                    "question_type": row.get("type"),
                    "supporting_facts": supporting_facts,
                    "evidences": evidences_raw,
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert 2WikiMultihopQA parquet into committee_llm task JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    count = convert(args.input, args.output, limit=args.limit)
    print(f"wrote {count} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
