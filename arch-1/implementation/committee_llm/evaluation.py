from __future__ import annotations

import re
import string
from collections import Counter
from statistics import mean
from typing import Any

from committee_llm.tasks import TaskSpec


def _word_count(text: str) -> int:
    return len(text.split())


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def _compute_exact(gold: str, predicted: str) -> float:
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


def _compute_f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()

    if len(gold_tokens) == 0 or len(predicted_tokens) == 0:
        return 1.0 if gold_tokens == predicted_tokens else 0.0

    common = Counter(gold_tokens) & Counter(predicted_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _metric_max_over_ground_truths(metric_fn: Any, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(ground_truth, prediction) for ground_truth in ground_truths)


def _strip_answer_header(answer: str) -> str:
    text = answer.strip()
    if not text:
        return text

    inline_prefixes = (
        "final answer:",
        "answer:",
        "core answer:",
        "short answer:",
        "draft deliverable:",
    )
    lowered = text.lower()
    for prefix in inline_prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    if lines[0].lower().rstrip(":") in {"final answer", "answer", "core answer", "short answer"}:
        remainder = "\n".join(lines[1:]).strip()
        if remainder:
            return remainder

    return text


def _candidate_answer_spans(answer: str) -> list[str]:
    text = _strip_answer_header(answer)
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = value.strip().rstrip(".!?;,")
        normalized = normalize_answer(cleaned)
        if not cleaned or not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(cleaned)

    add(text)

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    add(first_line)

    first_sentence = re.split(r"(?<=[.!?])\s+|\n", first_line or text, maxsplit=1)[0].strip()
    add(first_sentence)

    leading_clause = re.split(r"[,;]\s+|\s+(?:because|since|as)\s+", first_sentence, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    add(leading_clause)

    tokens = leading_clause.split()
    for prefix_length in range(1, min(len(tokens), 12) + 1):
        add(" ".join(tokens[:prefix_length]))

    return candidates or [answer.strip()]


def _best_reference_match(answer: str, ground_truths: list[str]) -> dict[str, Any]:
    best_candidate = answer.strip()
    best_em = _metric_max_over_ground_truths(_compute_exact, best_candidate, ground_truths)
    best_f1 = _metric_max_over_ground_truths(_compute_f1, best_candidate, ground_truths)
    best_key = (best_em, best_f1, -len(normalize_answer(best_candidate).split()))

    for candidate in _candidate_answer_spans(answer):
        exact = _metric_max_over_ground_truths(_compute_exact, candidate, ground_truths)
        f1 = _metric_max_over_ground_truths(_compute_f1, candidate, ground_truths)
        key = (exact, f1, -len(normalize_answer(candidate).split()))
        if key > best_key:
            best_candidate = candidate
            best_em = exact
            best_f1 = f1
            best_key = key

    return {
        "scored_answer": best_candidate,
        "answer_em": best_em,
        "answer_f1": best_f1,
    }


def _extract_choice_label(answer: str, choices: list[dict[str, str]]) -> str | None:
    if not choices:
        return None

    valid_labels = {choice["label"].upper() for choice in choices}
    patterns = [
        r"\banswer\s*[:\-]?\s*\(?([A-Z])\)?\b",
        r"\boption\s+\(?([A-Z])\)?\b",
        r"^\s*\(?([A-Z])\)?(?:[\.\):\-]|\s|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            label = match.group(1).upper()
            if label in valid_labels:
                return label

    normalized_answer = normalize_answer(answer)
    for choice in choices:
        normalized_choice = normalize_answer(choice["text"])
        if normalized_choice and (
            normalized_answer == normalized_choice
            or normalized_answer.startswith(normalized_choice + " ")
            or normalized_choice in normalized_answer
        ):
            return choice["label"].upper()

    return None


def evaluate_answer(task: TaskSpec, answer: str) -> dict[str, Any]:
    checks = task.checks
    normalized_answer = answer.lower()
    results: list[dict[str, Any]] = []

    for term in checks.get("must_contain", []):
        results.append(
            {
                "name": "must_contain",
                "target": term,
                "passed": term.lower() in normalized_answer,
            }
        )

    for term in checks.get("must_not_contain", []):
        results.append(
            {
                "name": "must_not_contain",
                "target": term,
                "passed": term.lower() not in normalized_answer,
            }
        )

    words = _word_count(answer)
    chars = len(answer)

    if "min_words" in checks:
        threshold = int(checks["min_words"])
        results.append({"name": "min_words", "target": threshold, "actual": words, "passed": words >= threshold})

    if "max_words" in checks:
        threshold = int(checks["max_words"])
        results.append({"name": "max_words", "target": threshold, "actual": words, "passed": words <= threshold})

    if "min_chars" in checks:
        threshold = int(checks["min_chars"])
        results.append({"name": "min_chars", "target": threshold, "actual": chars, "passed": chars >= threshold})

    if "max_chars" in checks:
        threshold = int(checks["max_chars"])
        results.append({"name": "max_chars", "target": threshold, "actual": chars, "passed": chars <= threshold})

    passed = None
    score = None
    if results:
        passed_flags = [bool(item["passed"]) for item in results]
        passed = all(passed_flags)
        score = mean(1.0 if flag else 0.0 for flag in passed_flags)

    reference_metrics: dict[str, float] = {}
    predicted_choice: str | None = None
    scored_answer: str | None = None

    if task.reference_answers:
        best_match = _best_reference_match(answer, task.reference_answers)
        scored_answer = str(best_match["scored_answer"])
        reference_metrics["answer_em"] = float(best_match["answer_em"])
        reference_metrics["answer_f1"] = float(best_match["answer_f1"])

    if task.correct_choice and task.choices:
        predicted_choice = _extract_choice_label(answer, task.choices)
        reference_metrics["choice_accuracy"] = 1.0 if predicted_choice == task.correct_choice else 0.0

    return {
        "domain": task.domain,
        "deliverable": task.deliverable,
        "evaluation_criteria": list(task.evaluation_criteria),
        "reference_count": len(task.references),
        "check_count": len(results),
        "passed": passed,
        "score": score,
        "answer_word_count": words,
        "answer_char_count": chars,
        "checks": results,
        "reference": {
            "benchmark_name": task.benchmark_name,
            "benchmark_split": task.benchmark_split,
            "task_type": task.task_type,
            "metrics": reference_metrics,
            "scored_answer": scored_answer,
            "predicted_choice": predicted_choice,
            "choice_count": len(task.choices),
            "reference_answer_count": len(task.reference_answers),
        },
    }
