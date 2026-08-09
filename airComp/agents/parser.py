"""Extract and validate a structured Offer from raw LLM text output."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from pydantic import BaseModel, ValidationError, field_validator

from airComp.env.negotiation import Offer, Pool, validate_offer_counts

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class OfferSchema(BaseModel):
    action: str
    counts: Optional[dict] = None
    message: Optional[str] = None

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v):
        if v not in ("propose", "accept", "reject"):
            raise ValueError(f"invalid action: {v!r}")
        return v


class ParseResult:
    def __init__(self, offer: Optional[Offer], error: Optional[str] = None):
        self.offer = offer
        self.error = error

    @property
    def ok(self) -> bool:
        return self.offer is not None


def extract_json_block(text: str) -> Optional[str]:
    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else None


def parse_offer(text: str, pool: Pool) -> ParseResult:
    block = extract_json_block(text)
    if block is None:
        return ParseResult(None, "no JSON object found in response")
    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        return ParseResult(None, f"invalid JSON: {e}")
    try:
        schema = OfferSchema.model_validate(data)
    except ValidationError as e:
        return ParseResult(None, f"schema validation failed: {e}")

    counts = None
    if schema.action == "propose":
        if not schema.counts:
            return ParseResult(None, "propose action is missing counts")
        try:
            counts = {t: int(schema.counts[t]) for t in pool.counts}
        except (KeyError, TypeError, ValueError):
            return ParseResult(None, "counts missing or non-integer for a required item type")
        if not validate_offer_counts(pool, counts):
            return ParseResult(None, "counts out of range for the pool")

    return ParseResult(Offer(action=schema.action, counts=counts, message=schema.message))


def parse_offer_with_retries(
    generate_fn: Callable[[int, Optional[str]], str],
    pool: Pool,
    max_retries: int = 2,
):
    """generate_fn(attempt_index, last_error) -> raw text from the LLM.

    Returns (offer_or_None, last_raw_text, attempts_used).
    """
    last_error = None
    raw_text = ""
    for attempt in range(max_retries + 1):
        raw_text = generate_fn(attempt, last_error)
        result = parse_offer(raw_text, pool)
        if result.ok:
            return result.offer, raw_text, attempt + 1
        last_error = result.error
    return None, raw_text, max_retries + 1
