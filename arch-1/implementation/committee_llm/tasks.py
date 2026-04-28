from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_CHECK_LIST_FIELDS = ("must_contain", "must_not_contain")
_CHECK_INT_FIELDS = ("min_words", "max_words", "min_chars", "max_chars")
_ALLOWED_CHECK_FIELDS = set(_CHECK_LIST_FIELDS + _CHECK_INT_FIELDS)
_VALID_TASK_TYPES = {"open_qa", "multiple_choice"}


def _read_optional_string(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string when provided.")

    text = value.strip()
    return text or None


def _read_string_list(raw: Mapping[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field '{key}' must be a list of strings.")

    items: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            raise ValueError(f"Field '{key}' item {index} must be a string.")
        text = item.strip()
        if text:
            items.append(text)
    return items


def _read_checks(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = raw.get("checks")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Field 'checks' must be an object.")

    unknown = sorted(set(value) - _ALLOWED_CHECK_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported check fields: {', '.join(unknown)}")

    checks: dict[str, Any] = {}

    for key in _CHECK_LIST_FIELDS:
        items = value.get(key)
        if items is None:
            continue
        if not isinstance(items, list):
            raise ValueError(f"Check '{key}' must be a list of strings.")
        normalized: list[str] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, str):
                raise ValueError(f"Check '{key}' item {index} must be a string.")
            text = item.strip()
            if text:
                normalized.append(text)
        if normalized:
            checks[key] = normalized

    for key in _CHECK_INT_FIELDS:
        number = value.get(key)
        if number is None:
            continue
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(f"Check '{key}' must be an integer.")
        if number < 0:
            raise ValueError(f"Check '{key}' must be non-negative.")
        checks[key] = number

    return checks


def _read_choices(raw: Mapping[str, Any]) -> list[dict[str, str]]:
    value = raw.get("choices")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Field 'choices' must be a list.")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices: list[dict[str, str]] = []

    for index, item in enumerate(value):
        if isinstance(item, str):
            text = item.strip()
            label = labels[index]
        elif isinstance(item, dict):
            raw_label = item.get("label", labels[index])
            raw_text = item.get("text", item.get("option", item.get("content")))
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError(f"Choice {index + 1} must have a non-empty string label.")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(f"Choice {index + 1} must have a non-empty string text.")
            label = raw_label.strip().upper()
            text = raw_text.strip()
        else:
            raise ValueError(f"Choice {index + 1} must be a string or object.")

        if not text:
            raise ValueError(f"Choice {index + 1} text must not be empty.")
        choices.append({"label": label, "text": text})

    labels_seen = [choice["label"] for choice in choices]
    if len(labels_seen) != len(set(labels_seen)):
        raise ValueError("Choice labels must be unique.")

    return choices


@dataclass(slots=True)
class TaskSpec:
    prompt: str
    task_id: str | None = None
    domain: str | None = None
    benchmark_name: str | None = None
    benchmark_split: str | None = None
    task_type: str | None = None
    context: str | None = None
    objectives: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    deliverable: str | None = None
    evaluation_criteria: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    reference_answers: list[str] = field(default_factory=list)
    choices: list[dict[str, str]] = field(default_factory=list)
    correct_choice: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TaskSpec":
        text = prompt.strip()
        if not text:
            raise ValueError("Task prompt must not be empty.")
        return cls(prompt=text, task_id=task_id, metadata=dict(metadata or {}))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, default_task_id: str | None = None) -> "TaskSpec":
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Field 'prompt' is required and must be a non-empty string.")

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Field 'metadata' must be an object when provided.")

        task_id = _read_optional_string(raw, "id") or default_task_id
        benchmark_name = _read_optional_string(raw, "benchmark_name") or _read_optional_string(raw, "benchmark")
        benchmark_split = _read_optional_string(raw, "benchmark_split")
        task_type = _read_optional_string(raw, "task_type")
        if task_type is not None and task_type not in _VALID_TASK_TYPES:
            raise ValueError(f"Field 'task_type' must be one of: {', '.join(sorted(_VALID_TASK_TYPES))}")

        reference_answers = _read_string_list(raw, "reference_answers")
        if not reference_answers:
            reference_answers = _read_string_list(raw, "gold_answers")

        choices = _read_choices(raw)
        correct_choice = _read_optional_string(raw, "correct_choice")
        if correct_choice is not None:
            correct_choice = correct_choice.upper()
            if not choices:
                raise ValueError("Field 'correct_choice' requires 'choices'.")
            choice_labels = {choice["label"] for choice in choices}
            if correct_choice not in choice_labels:
                raise ValueError("Field 'correct_choice' must match one of the choice labels.")

        return cls(
            prompt=prompt.strip(),
            task_id=task_id,
            domain=_read_optional_string(raw, "domain"),
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
            task_type=task_type,
            context=_read_optional_string(raw, "context"),
            objectives=_read_string_list(raw, "objectives"),
            constraints=_read_string_list(raw, "constraints"),
            deliverable=_read_optional_string(raw, "deliverable"),
            evaluation_criteria=_read_string_list(raw, "evaluation_criteria"),
            references=_read_string_list(raw, "references"),
            reference_answers=reference_answers,
            choices=choices,
            correct_choice=correct_choice,
            checks=_read_checks(raw),
            metadata=dict(metadata),
        )

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "prompt": self.prompt,
            "domain": self.domain,
            "benchmark_name": self.benchmark_name,
            "benchmark_split": self.benchmark_split,
            "task_type": self.task_type,
            "context": self.context,
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
            "deliverable": self.deliverable,
            "evaluation_criteria": list(self.evaluation_criteria),
            "references": list(self.references),
            "reference_answers": list(self.reference_answers),
            "choices": [dict(choice) for choice in self.choices],
            "correct_choice": self.correct_choice,
            "checks": dict(self.checks),
            "metadata": dict(self.metadata),
        }
