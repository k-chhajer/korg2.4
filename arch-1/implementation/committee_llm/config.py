from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROLE_SEQUENCE = (
    "coordinator_plan",
    "researcher",
    "analyst",
    "critic",
    "coordinator_finalize",
)


@dataclass(slots=True)
class ProviderConfig:
    api_base: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_sec: int = 180


@dataclass(slots=True)
class DefaultRoleSettings:
    model: str
    temperature: float = 0.3
    max_tokens: int = 1200


@dataclass(slots=True)
class RoleConfig:
    system_prompt_path: Path
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    provider: ProviderConfig
    defaults: DefaultRoleSettings
    roles: dict[str, RoleConfig]
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source_path = Path(path).resolve()
        raw = json.loads(source_path.read_text(encoding="utf-8"))

        provider = ProviderConfig(**raw["provider"])
        defaults = DefaultRoleSettings(**raw["defaults"])

        roles: dict[str, RoleConfig] = {}
        for role_name in ROLE_SEQUENCE:
            if role_name not in raw["roles"]:
                raise ValueError(f"Missing required role '{role_name}' in {source_path}")

            role_raw = dict(raw["roles"][role_name])
            prompt_path = Path(role_raw.pop("system_prompt_path"))
            if not prompt_path.is_absolute():
                prompt_path = (source_path.parent / prompt_path).resolve()

            roles[role_name] = RoleConfig(system_prompt_path=prompt_path, **role_raw)

        return cls(
            name=raw["name"],
            provider=provider,
            defaults=defaults,
            roles=roles,
            source_path=source_path,
        )

    def resolve_role_settings(self, role_name: str) -> dict[str, str | float | int]:
        role = self.roles[role_name]
        return {
            "model": role.model or self.defaults.model,
            "temperature": role.temperature
            if role.temperature is not None
            else self.defaults.temperature,
            "max_tokens": role.max_tokens
            if role.max_tokens is not None
            else self.defaults.max_tokens,
        }

    def load_system_prompt(self, role_name: str) -> str:
        prompt_path = self.roles[role_name].system_prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8").strip()
