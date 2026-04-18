"""Tests for RLVR reward function - no shaped components."""

from committee_llm.adaptive import compute_verifiable_reward


def test_gpqa_exact_match() -> None:
    assert compute_verifiable_reward("A", "A", "gpqa") == 1.0
    assert compute_verifiable_reward("B", "A", "gpqa") == 0.0


def test_gpqa_letter_extraction() -> None:
    assert compute_verifiable_reward("A) Photosynthesis occurs in chloroplasts", "A", "gpqa") == 1.0


def test_hotpotqa_exact() -> None:
    assert compute_verifiable_reward("Marie Curie", "Marie Curie", "hotpotqa") == 1.0


def test_hotpotqa_partial_f1() -> None:
    reward = compute_verifiable_reward("Marie Curie was a physicist", "Marie Curie", "hotpotqa")
    assert 0.0 < reward < 1.0


def test_hotpotqa_wrong() -> None:
    assert compute_verifiable_reward("Albert Einstein", "Marie Curie", "hotpotqa") == 0.0
