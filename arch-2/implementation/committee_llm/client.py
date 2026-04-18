from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass(slots=True)
class ChatResult:
    model: str
    content: str
    elapsed_sec: float
    usage: dict[str, Any] | None
    raw_response: dict[str, Any]


class OpenAICompatibleChatClient:
    def __init__(self, api_base: str, api_key_env: str = "OPENAI_API_KEY", timeout_sec: int = 180):
        api_base = api_base.rstrip("/")
        if api_base.endswith("/v1"):
            self.endpoint = f"{api_base}/chat/completions"
        else:
            self.endpoint = f"{api_base}/v1/chat/completions"

        self._load_dotenv_if_needed(api_key_env)
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.timeout_sec = timeout_sec

    @staticmethod
    def _load_dotenv_if_needed(api_key_env: str) -> None:
        # Keep explicit environment variables highest priority.
        if os.environ.get(api_key_env):
            return

        # Search .env from current working directory upward (repo-root friendly).
        current = Path.cwd().resolve()
        candidates: list[Path] = []
        candidates.extend(current.parents)
        candidates.insert(0, current)

        for directory in candidates:
            env_path = directory / ".env"
            if not env_path.exists():
                continue
            try:
                lines = env_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue

            for raw_line in lines:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (
                    len(value) >= 2
                    and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
                ):
                    value = value[1:-1]
                if key not in os.environ:
                    os.environ[key] = value

            if os.environ.get(api_key_env):
                return

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        http_request = request.Request(self.endpoint, data=body, headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with request.urlopen(http_request, timeout=self.timeout_sec) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Chat request failed with HTTP {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach chat backend at {self.endpoint}: {exc.reason}"
            ) from exc

        elapsed_sec = time.perf_counter() - start

        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected chat response format from backend: {raw_response}"
            ) from exc

        return ChatResult(
            model=raw_response.get("model", model),
            content=content,
            elapsed_sec=elapsed_sec,
            usage=raw_response.get("usage"),
            raw_response=raw_response,
        )
