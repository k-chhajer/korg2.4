from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def merge_numeric_dicts(target: dict[str, Any], source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not source:
        return target

    for key, value in source.items():
        if _is_numeric(value):
            current = target.get(key, 0)
            target[key] = (current if _is_numeric(current) else 0) + value
            continue

        if isinstance(value, Mapping):
            nested = target.get(key)
            if not isinstance(nested, dict):
                nested = {}
                target[key] = nested
            merge_numeric_dicts(nested, value)

    return target


def aggregate_usage(usages: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for usage in usages:
        merge_numeric_dicts(total, usage)
    return total
