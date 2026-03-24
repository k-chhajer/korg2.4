from __future__ import annotations

import json
from typing import Any


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def parse_json_from_text(text: str) -> Any:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no_json_payload_found")
