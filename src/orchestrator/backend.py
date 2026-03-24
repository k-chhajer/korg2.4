from __future__ import annotations

import json
import os
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
def build_backend(runtime_config: RuntimeConfig) -> Backend:
    provider = str(runtime_config.backend.get("provider", "openrouter")).lower()
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

