from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.tools.json_utils import parse_json_from_text
from src.tools.token_utils import count_tokens_approx


@dataclass
class HarperValidationResult:
    accepted: bool
    payload: dict[str, Any]
    errors: list[str]
    token_count: int


def validate_harper_output(
    raw: str,
    valid_source_ids: set[int],
    max_claims: int,
    max_open_questions: int,
    max_tokens: int,
) -> HarperValidationResult:
    errors: list[str] = []
    token_count = count_tokens_approx(raw)
    if token_count > max_tokens:
        errors.append("harper_output_too_long")

    try:
        payload = parse_json_from_text(raw)
    except ValueError:
        return HarperValidationResult(
            accepted=False,
            payload={},
            errors=["invalid_json"],
            token_count=token_count,
        )

    if not isinstance(payload, dict):
        return HarperValidationResult(
            accepted=False,
            payload={},
            errors=["harper_output_not_object"],
            token_count=token_count,
        )

    required = {"answer_summary", "claims", "source_digest", "open_questions"}
    missing = required - set(payload.keys())
    if missing:
        errors.append(f"missing_keys:{sorted(missing)}")

    answer_summary = payload.get("answer_summary")
    claims = payload.get("claims")
    source_digest = payload.get("source_digest")
    open_questions = payload.get("open_questions")

    if not isinstance(answer_summary, str) or not answer_summary.strip():
        errors.append("invalid_answer_summary")

    if not isinstance(claims, list) or not claims:
        errors.append("invalid_claims")
        claims = []
    if len(claims) > max_claims:
        errors.append("too_many_claims")

    if not isinstance(source_digest, list) or not source_digest:
        errors.append("invalid_source_digest")
        source_digest = []

    if not isinstance(open_questions, list):
        errors.append("invalid_open_questions")
        open_questions = []
    if len(open_questions) > max_open_questions:
        errors.append("too_many_open_questions")

    allowed_support = {"supported", "partial", "uncertain", "contradicted"}
    cited_source_ids: set[int] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("claim_not_object")
            continue
        required_claim = {"claim", "support", "confidence", "sources", "notes"}
        missing_claim = required_claim - set(claim.keys())
        if missing_claim:
            errors.append(f"claim_missing_keys:{sorted(missing_claim)}")
            continue

        if not isinstance(claim["claim"], str) or not claim["claim"].strip():
            errors.append("claim_text_invalid")
        if claim["support"] not in allowed_support:
            errors.append("claim_support_invalid")
        try:
            confidence = float(claim["confidence"])
        except (TypeError, ValueError):
            errors.append("claim_confidence_invalid")
            confidence = -1.0
        if confidence < 0.0 or confidence > 1.0:
            errors.append("claim_confidence_out_of_range")

        if not isinstance(claim["sources"], list):
            errors.append("claim_sources_invalid")
            continue
        for source_id in claim["sources"]:
            if not isinstance(source_id, int):
                errors.append("claim_source_id_not_int")
                continue
            if source_id not in valid_source_ids:
                errors.append("claim_source_id_unknown")
            cited_source_ids.add(source_id)

        if not isinstance(claim["notes"], str):
            errors.append("claim_notes_invalid")

    digest_source_ids: set[int] = set()
    for digest in source_digest:
        if not isinstance(digest, dict):
            errors.append("source_digest_item_not_object")
            continue
        required_digest = {"source_id", "title", "url", "why_it_matters"}
        missing_digest = required_digest - set(digest.keys())
        if missing_digest:
            errors.append(f"source_digest_missing_keys:{sorted(missing_digest)}")
            continue

        source_id = digest["source_id"]
        if not isinstance(source_id, int):
            errors.append("source_digest_id_not_int")
            continue
        if source_id not in valid_source_ids:
            errors.append("source_digest_id_unknown")
        digest_source_ids.add(source_id)
        if not isinstance(digest["title"], str) or not digest["title"].strip():
            errors.append("source_digest_title_invalid")
        if not isinstance(digest["url"], str) or not digest["url"].strip():
            errors.append("source_digest_url_invalid")
        if not isinstance(digest["why_it_matters"], str):
            errors.append("source_digest_why_invalid")

    if cited_source_ids - digest_source_ids:
        errors.append("cited_source_missing_from_digest")

    normalized = {}
    if not errors:
        normalized = {
            "answer_summary": answer_summary.strip(),
            "claims": claims,
            "source_digest": source_digest,
            "open_questions": [str(item).strip() for item in open_questions if str(item).strip()],
        }

    return HarperValidationResult(
        accepted=len(errors) == 0,
        payload=normalized,
        errors=errors,
        token_count=token_count,
    )
