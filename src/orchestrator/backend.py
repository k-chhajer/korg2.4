from __future__ import annotations

import hashlib
import json
import random
from typing import Protocol


class Backend(Protocol):
    def generate(
        self,
        role: str,
        stage: str,
        prompt: str,
        max_tokens: int,
        seed: int,
    ) -> str:
        ...


class MockBackend:
    """
    Deterministic backend for Milestone 0/1 integration and eval harness checks.
    """

    def _rng(self, role: str, stage: str, prompt: str, seed: int) -> random.Random:
        digest = hashlib.sha256(f"{role}|{stage}|{prompt}|{seed}".encode("utf-8")).hexdigest()
        seed_int = int(digest[:12], 16)
        return random.Random(seed_int)

    def generate(
        self,
        role: str,
        stage: str,
        prompt: str,
        max_tokens: int,
        seed: int,
    ) -> str:
        rng = self._rng(role, stage, prompt, seed)
        query_preview = prompt.splitlines()[-1][:80] if prompt else "query"

        if stage == "captain_plan":
            return (
                "1) Goal: deliver a coherent answer.\n"
                "2) Harper task: extract high-confidence facts and evidence gaps.\n"
                "3) Benjamin task: validate logic and edge-case consistency.\n"
                "4) Lucas task: generate alternatives and UX tradeoffs.\n"
                "5) Conflict checks: resolve factual-vs-logical mismatches explicitly."
            )

        if stage == "role_output":
            if role == "harper":
                return (
                    f"facts: grounded points for '{query_preview}'.\n"
                    "evidence_gaps: missing source timestamps and uncertainty labels.\n"
                    "confidence: medium-high with explicit caveats."
                )
            if role == "benjamin":
                return (
                    f"logic: derive constraints and validate assumptions for '{query_preview}'.\n"
                    "risks: hidden coupling, brittle invariants, and silent failures.\n"
                    "checks: add deterministic tests and contradiction checks."
                )
            return (
                f"alternative: offer a second path for '{query_preview}'.\n"
                "ux_considerations: reduce cognitive load and simplify outputs.\n"
                "tradeoffs: slight latency increase for better clarity and adoption."
            )

        if stage == "critique":
            challenge_type = rng.choice(
                ["factual", "logical", "missing_consideration", "alternative_approach"]
            )
            payload = {
                "challenge_type": challenge_type,
                "target": "confidence labels are implicit",
                "correction": "add explicit confidence per claim with one-line rationale",
            }
            return json.dumps(payload)

        if stage == "captain_synthesis":
            return (
                "facts: align outputs with explicit evidence confidence and assumptions.\n"
                "logic: enforce testable constraints, contradiction checks, and fallback paths.\n"
                "alternative: include one low-risk rollout path and one faster experimental path.\n"
                "final recommendation: execute phased delivery with validation gates per phase."
            )

        return "no-op"

