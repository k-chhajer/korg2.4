from __future__ import annotations


def count_tokens_approx(text: str) -> int:
    """Cheap token proxy for deterministic local eval."""
    if not text:
        return 0
    return len(text.strip().split())

