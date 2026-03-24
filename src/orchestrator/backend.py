from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from typing import Protocol

from src.orchestrator.types import RuntimeConfig


class Backend(Protocol):
    def generate(
        self,
        role: str,
        stage: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        seed: int,
        temperature: float,
    ) -> str:
        ...


class MockBackend:
    """
    Deterministic backend for Milestone 1 integration and eval harness checks.
    """

    def _rng(self, role: str, stage: str, prompt: str, seed: int) -> random.Random:
        digest = hashlib.sha256(f"{role}|{stage}|{prompt}|{seed}".encode("utf-8")).hexdigest()
        seed_int = int(digest[:12], 16)
        return random.Random(seed_int)

    def generate(
        self,
        role: str,
        stage: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        seed: int,
        temperature: float,
    ) -> str:
        rng = self._rng(role, stage, prompt, seed)
        query_preview = _extract_user_query(prompt)

        if stage == "captain_plan":
            return json.dumps(
                {
                    "overall_strategy": (
                        "Split work into grounded research, independent reasoning, "
                        "and a review pass before synthesis."
                    ),
                    "harper_task": (
                        "Search for recent, high-signal facts and implementation patterns "
                        "that matter for the user query."
                    ),
                    "benjamin_task": (
                        "Stress-test the architecture, identify edge cases, and produce a "
                        "defensible implementation path."
                    ),
                    "lucas_task": (
                        "Review Harper and Benjamin, identify gaps, add human-centered framing, "
                        "and tighten the recommendation."
                    ),
                    "synthesis_checks": [
                        "Separate evidence from inference.",
                        "Call out missing infrastructure or API requirements.",
                        "Produce a practical first iteration, not an idealized design.",
                    ],
                    "deliverable_shape": "practical implementation guidance with explicit next steps",
                },
                indent=2,
            )

        if stage == "harper_search_plan":
            return json.dumps(
                [
                    f"{query_preview} architecture examples",
                    f"{query_preview} Qwen OpenRouter setup",
                    f"{query_preview} web search integration patterns",
                ],
                indent=2,
            )

        if stage == "harper_research":
            return (
                "summary: Web research suggests using a hosted chat-completions backend and a "
                "separate search provider rather than trying to force both into one layer.\n"
                "evidence: [1] Hosted Qwen inference can sit behind a generic OpenAI-compatible API. "
                "[2] Search works best when query generation and result grounding are explicit steps.\n"
                "open_questions: Confirm which search provider and API budget are acceptable.\n"
                "source_notes: Prefer provider APIs over ad-hoc scraping when reliability matters."
            )

        if stage == "benjamin_reasoning":
            return (
                "logic: The lowest-risk first iteration is a sequential pipeline with captain planning, "
                "Harper search, Benjamin analysis, Lucas review, and captain synthesis.\n"
                "assumptions: External APIs may fail or rate-limit, so tool calls need fallbacks and "
                "timeouts.\n"
                "failure_modes: search drift, citation loss, weak decomposition, and overlong prompts.\n"
                "checks: enforce structured captain plans, cap search queries, and keep the backend "
                "interface provider-agnostic."
            )

        if stage == "lucas_review":
            return (
                "critique: Harper adds evidence, but Benjamin should define the decision boundary more "
                "explicitly when evidence is thin.\n"
                "user_impact: The answer should tell the user what keys they need and what still works "
                "without them.\n"
                "communication_advice: Lead with the working architecture, then list optional upgrades.\n"
                "recommendation: Ship a robust brute-force version first and improve latency after the "
                "tooling loop is trustworthy."
            )

        if stage == "lucas_critique":
            challenge_type = rng.choice(
                ["factual", "logical", "missing_consideration", "alternative_approach"]
            )
            payload = {
                "challenge_type": challenge_type,
                "target": "search provider is underspecified",
                "correction": "state required API keys and fallback behavior explicitly",
            }
            return json.dumps(payload)

        if stage == "captain_synthesis":
            return (
                "facts: Use a provider-backed Qwen runtime for generation and an explicit search tool for "
                "Harper instead of collapsing both concerns into one backend.\n"
                "logic: Keep the first version sequential with strong structure, fallbacks, and tool call "
                "timeouts before optimizing latency.\n"
                "alternative: If you do not want a paid search API yet, start with DuckDuckGo as a "
                "best-effort fallback and move to Serper once reliability matters.\n"
                "final recommendation: build the captain-harper-benjamin-lucas-captain loop now, wire "
                "it to OpenRouter for Qwen, and treat search reliability as the first external dependency."
            )

        return "no-op"


class OpenRouterBackend:
    def __init__(
        self,
        model: str,
        timeout_seconds: int = 60,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "grok_multiagent")
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when backend.provider=openrouter")

    def generate(
        self,
        role: str,
        stage: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        seed: int,
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"openrouter_http_error:{exc.code}:{details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"openrouter_network_error:{exc.reason}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"openrouter_bad_response:{body}") from exc

        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)


def _extract_user_query(prompt: str) -> str:
    marker = "User query:\n"
    if marker in prompt:
        return prompt.split(marker, maxsplit=1)[1].strip().splitlines()[0][:80]
    return prompt.splitlines()[-1][:80] if prompt else "query"


def build_backend(runtime_config: RuntimeConfig) -> Backend:
    provider = str(runtime_config.backend.get("provider", "mock")).lower()
    if provider == "mock":
        return MockBackend()
    if provider == "openrouter":
        return OpenRouterBackend(
            model=str(runtime_config.backend.get("model", "qwen/qwen3-14b")),
            timeout_seconds=int(runtime_config.backend.get("timeout_seconds", 60)),
            base_url=str(
                runtime_config.backend.get(
                    "base_url", "https://openrouter.ai/api/v1/chat/completions"
                )
            ),
            site_url=str(runtime_config.backend.get("site_url", "http://localhost")),
            app_name=str(runtime_config.backend.get("app_name", "grok_multiagent")),
        )
    raise ValueError(f"unsupported backend provider: {provider}")

