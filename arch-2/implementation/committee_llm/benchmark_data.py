from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _limit_items(items: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(items)
    return list(items)[:limit]


def _format_context_paragraphs(paragraphs: list[tuple[str, str]]) -> str:
    formatted = []
    for index, (title, text) in enumerate(paragraphs, start=1):
        clean_text = " ".join(text.split())
        formatted.append(f"[{index}] {title}\n{clean_text}")
    return "\n\n".join(formatted)


def convert_hotpotqa(input_path: str | Path, output_path: str | Path, *, limit: int | None = None) -> int:
    raw_items = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError("HotpotQA input must be a JSON array.")

    converted: list[dict[str, Any]] = []
    for item in _limit_items(raw_items, limit):
        context_items = item.get("context", [])
        paragraphs: list[tuple[str, str]] = []
        titles: list[str] = []
        for paragraph in context_items:
            if not isinstance(paragraph, list) or len(paragraph) != 2:
                continue
            title, sentences = paragraph
            if not isinstance(title, str) or not isinstance(sentences, list):
                continue
            text = " ".join(sentence.strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip())
            paragraphs.append((title, text))
            titles.append(title)

        converted.append(
            {
                "id": str(item.get("_id")),
                "benchmark_name": "hotpotqa_distractor",
                "benchmark_split": "dev",
                "task_type": "open_qa",
                "domain": "multi_hop_qa",
                "prompt": str(item["question"]).strip(),
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
                "reference_answers": [str(item["answer"]).strip()],
                "metadata": {
                    "source_dataset": "HotpotQA",
                    "question_type": item.get("type"),
                    "supporting_facts": item.get("supporting_facts", []),
                },
            }
        )

    _write_jsonl(Path(output_path), converted)
    return len(converted)


def convert_2wikimultihop(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    split: str | None = None,
) -> int:
    raw_items = _read_json_items(input_path)
    converted: list[dict[str, Any]] = []

    for item in _limit_items(raw_items, limit):
        context_items = item.get("context", [])
        paragraphs: list[tuple[str, str]] = []
        titles: list[str] = []
        for paragraph in context_items:
            if not isinstance(paragraph, list) or len(paragraph) != 2:
                continue
            title, sentences = paragraph
            if not isinstance(title, str) or not isinstance(sentences, list):
                continue
            text = " ".join(sentence.strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip())
            if not text:
                continue
            paragraphs.append((title, text))
            titles.append(title)

        answer = str(item.get("answer", "")).strip()
        if not answer:
            continue

        question_type = str(item.get("type") or "").strip()
        task_id = str(item.get("_id") or item.get("id") or f"2wiki-{len(converted) + 1:05d}")

        converted.append(
            {
                "id": task_id,
                "benchmark_name": "2wikimultihop",
                "benchmark_split": split or str(item.get("split") or "train"),
                "task_type": "open_qa",
                "domain": "multi_hop_qa",
                "prompt": str(item["question"]).strip(),
                "context": _format_context_paragraphs(paragraphs),
                "deliverable": (
                    "Answer the multi-hop question using only the provided context. "
                    "Start with a short answer phrase, then briefly justify the evidence chain."
                ),
                "evaluation_criteria": [
                    "Answer should match the gold answer under normalized EM/F1.",
                    "Reasoning should use the provided context and connect the relevant hops.",
                ],
                "references": titles,
                "reference_answers": [answer],
                "metadata": {
                    "source_dataset": "2WikiMultihopQA",
                    "question_type": question_type or None,
                    "supporting_facts": item.get("supporting_facts", []),
                    "evidences": item.get("evidences", []),
                    "evidences_id": item.get("evidences_id", []),
                    "entity_ids": item.get("entity_ids"),
                    "answer_id": item.get("answer_id"),
                },
            }
        )

    _write_jsonl(Path(output_path), converted)
    return len(converted)


def _read_json_items(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("Expected a JSON array.")
        return [item for item in raw if isinstance(item, dict)]

    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("Expected each JSONL line to be an object.")
        items.append(raw)
    return items


def convert_musique(input_path: str | Path, output_path: str | Path, *, limit: int | None = None) -> int:
    raw_items = _read_json_items(input_path)
    converted: list[dict[str, Any]] = []

    for item in _limit_items(raw_items, limit):
        paragraphs_raw = item.get("paragraphs", [])
        paragraphs: list[tuple[str, str]] = []
        titles: list[str] = []
        for index, paragraph in enumerate(paragraphs_raw, start=1):
            if not isinstance(paragraph, dict):
                continue
            title = str(paragraph.get("title") or paragraph.get("wikipedia_title") or f"Paragraph {index}")
            text = str(
                paragraph.get("paragraph_text")
                or paragraph.get("paragraph")
                or paragraph.get("text")
                or paragraph.get("context")
                or ""
            ).strip()
            if not text:
                continue
            paragraphs.append((title, text))
            titles.append(title)

        answer = str(item.get("answer", "")).strip()
        answer_aliases = [
            alias.strip() for alias in item.get("answer_aliases", []) if isinstance(alias, str) and alias.strip()
        ]
        reference_answers = [candidate for candidate in [answer] + answer_aliases if candidate]

        converted.append(
            {
                "id": str(item.get("id") or item.get("_id")),
                "benchmark_name": "musique_ans",
                "benchmark_split": str(item.get("split", "dev")),
                "task_type": "open_qa",
                "domain": "multi_hop_qa",
                "prompt": str(item["question"]).strip(),
                "context": _format_context_paragraphs(paragraphs),
                "deliverable": (
                    "Answer the question using the provided paragraphs. "
                    "Start with a short answer phrase, then briefly justify it."
                ),
                "evaluation_criteria": [
                    "Answer should match one of the accepted answers under normalized EM/F1.",
                    "Reasoning should use the provided paragraphs.",
                ],
                "references": titles,
                "reference_answers": reference_answers,
                "metadata": {
                    "source_dataset": "MuSiQue",
                    "answerable": item.get("answerable", True),
                    "question_decomposition": item.get("question_decomposition", []),
                },
            }
        )

    _write_jsonl(Path(output_path), converted)
    return len(converted)


def _pick_field(row: dict[str, str], candidates: list[str]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def convert_gpqa(input_path: str | Path, output_path: str | Path, *, limit: int | None = None) -> int:
    converted: list[dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    for index, row in enumerate(rows[:limit] if limit is not None else rows, start=1):
        question = _pick_field(row, ["Question", "question"])
        correct_answer = _pick_field(row, ["Correct Answer", "correct_answer", "Correct answer"])
        incorrect_answers = [
            _pick_field(row, [f"Incorrect Answer {offset}", f"incorrect_answer_{offset}", f"Incorrect answer {offset}"])
            for offset in range(1, 4)
        ]
        if not question or not correct_answer or any(answer is None for answer in incorrect_answers):
            raise ValueError("GPQA rows must contain one question, one correct answer, and three incorrect answers.")

        option_pool = [{"text": correct_answer, "is_correct": True}] + [
            {"text": str(answer), "is_correct": False} for answer in incorrect_answers
        ]
        seed_basis = _pick_field(row, ["Question ID", "id", "ID"]) or question
        rng = random.Random(seed_basis)
        rng.shuffle(option_pool)

        labels = ["A", "B", "C", "D"]
        choices = []
        correct_choice = None
        for label, option in zip(labels, option_pool):
            choices.append({"label": label, "text": option["text"]})
            if option["is_correct"]:
                correct_choice = label

        converted.append(
            {
                "id": _pick_field(row, ["Question ID", "id", "ID"]) or f"gpqa-{index:05d}",
                "benchmark_name": "gpqa",
                "benchmark_split": _pick_field(row, ["Split", "split"]) or "eval",
                "task_type": "multiple_choice",
                "domain": _pick_field(row, ["High-level domain", "high_level_domain", "Domain"]) or "science_qa",
                "prompt": question,
                "choices": choices,
                "correct_choice": correct_choice,
                "deliverable": (
                    "Select exactly one answer choice. Start the response with `Answer: <label>` "
                    "and then briefly justify the choice."
                ),
                "evaluation_criteria": [
                    "The selected label should match the correct answer choice.",
                ],
                "metadata": {
                    "source_dataset": "GPQA",
                    "subdomain": _pick_field(row, ["Subdomain", "subdomain"]),
                },
            }
        )

    _write_jsonl(Path(output_path), converted)
    return len(converted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert official benchmark data into committee_llm task JSONL.")
    parser.add_argument("--suite", required=True, choices=["hotpotqa", "musique", "gpqa", "2wikimultihop"])
    parser.add_argument("--input", required=True, help="Path to the original benchmark file.")
    parser.add_argument("--output", required=True, help="Path to the converted JSONL output.")
    parser.add_argument("--limit", type=int, help="Optional max number of examples to convert.")
    parser.add_argument("--split", help="Optional split name to store in converted task metadata.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.suite == "hotpotqa":
        count = convert_hotpotqa(args.input, args.output, limit=args.limit)
    elif args.suite == "musique":
        count = convert_musique(args.input, args.output, limit=args.limit)
    elif args.suite == "2wikimultihop":
        count = convert_2wikimultihop(args.input, args.output, limit=args.limit, split=args.split)
    else:
        count = convert_gpqa(args.input, args.output, limit=args.limit)

    print(f"wrote {count} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
