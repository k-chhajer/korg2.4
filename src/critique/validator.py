from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.tools.token_utils import count_tokens_approx


@dataclass
class CritiqueValidationResult:
    accepted: bool
    critique: dict[str, str]
    errors: list[str]
    token_count: int


def validate_critique(
    raw: str,
    allowed_types: set[str],
    target_max_tokens: int,
    correction_max_tokens: int,
    critique_max_tokens: int,
) -> CritiqueValidationResult:
    errors: list[str] = []
    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid_json:{exc.msg}")
        return CritiqueValidationResult(
            accepted=False,
            critique={},
            errors=errors,
            token_count=0,
        )

    required = {"challenge_type", "target", "correction"}
    extra = set(parsed.keys()) - required
    missing = required - set(parsed.keys())
    if missing:
        errors.append(f"missing_keys:{sorted(missing)}")
    if extra:
        errors.append(f"extra_keys:{sorted(extra)}")

    if not missing:
        challenge_type = str(parsed["challenge_type"])
        target = str(parsed["target"])
        correction = str(parsed["correction"])

        if challenge_type not in allowed_types:
            errors.append("invalid_challenge_type")

        target_tokens = count_tokens_approx(target)
        correction_tokens = count_tokens_approx(correction)
        total_tokens = count_tokens_approx(raw)

        if target_tokens > target_max_tokens:
            errors.append("target_too_long")
        if correction_tokens > correction_max_tokens:
            errors.append("correction_too_long")
        if total_tokens > critique_max_tokens:
            errors.append("critique_too_long")
    else:
        total_tokens = count_tokens_approx(raw)

    accepted = len(errors) == 0
    critique = {}
    if accepted:
        critique = {
            "challenge_type": str(parsed["challenge_type"]),
            "target": str(parsed["target"]),
            "correction": str(parsed["correction"]),
        }
    return CritiqueValidationResult(
        accepted=accepted,
        critique=critique,
        errors=errors,
        token_count=total_tokens,
    )


def validate_total_critique_budget(
    critiques: list[dict[str, str]],
    total_critique_budget: int,
) -> bool:
    total = sum(count_tokens_approx(json.dumps(c)) for c in critiques)
    return total <= total_critique_budget

