# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates shared by TextAgent and SemanticAgent.

Both pipelines reuse `history_prompt` to describe prior turns to the LLM as
natural language, regardless of whether a turn's `received_offer` came from
the digital channel (baseline) or the JSCC decoder (semantic) -- from the
LLM's point of view it is just "what the other agent did."
"""
from __future__ import annotations

from typing import Optional

from airComp.env.negotiation import Offer, Pool, Values

OFFER_JSON_INSTRUCTIONS = (
    "You must respond with a single JSON object and nothing else, matching this schema:\n"
    '{"action": "propose" | "accept" | "reject", '
    '"counts": {"book": <int>, "hat": <int>, "ball": <int>}, '
    '"message": "<short optional one-sentence rationale>"}\n'
    'The "counts" field is only used for "propose" and means the amount YOU want to keep '
    '(the rest goes to the other agent). "accept" and "reject" do not need a "counts" field.'
)


def system_prompt(pool: Pool, own_values: Values, max_messages: int, include_rationale: bool = True) -> str:
    pool_str = ", ".join(f"{count} {t}" for t, count in pool.counts.items())
    value_str = ", ".join(f"{t}={v:.1f} pts/unit" for t, v in own_values.per_unit.items())
    rationale_note = "" if include_rationale else '\nOmit the "message" field or leave it empty.'
    return (
        "You are negotiating with another agent over how to split a shared pool of items.\n"
        f"Pool: {pool_str}.\n"
        f"Your private per-unit values: {value_str}. Taking the whole pool would be worth "
        "100 points to you.\n"
        "The other agent has its own private values, which you do not know.\n"
        f"You have at most {max_messages} total messages (both agents combined) to reach an "
        "agreement. If no agreement is reached, you both get 0 points.\n\n"
        f"{OFFER_JSON_INSTRUCTIONS}{rationale_note}"
    )


def history_prompt(history: list, standing_offer: Optional[Offer]) -> str:
    if not history:
        return "No messages yet. Make the first proposal."
    lines = []
    for turn in history:
        offer = turn.received_offer
        if offer is None:
            lines.append(f"Turn {turn.turn_index} ({turn.agent}): [message lost / unparseable]")
        elif offer.action == "propose":
            counts_str = ", ".join(f"{v} {t}" for t, v in (offer.counts or {}).items())
            lines.append(f"Turn {turn.turn_index} ({turn.agent}): proposes to keep {counts_str}.")
        else:
            lines.append(f"Turn {turn.turn_index} ({turn.agent}): {offer.action}.")
    if standing_offer is not None:
        counts_str = ", ".join(f"{v} {t}" for t, v in (standing_offer.counts or {}).items())
        lines.append(
            f"Standing offer on the table: proposer keeps {counts_str}. "
            "You may ACCEPT, counter-PROPOSE, or REJECT."
        )
    return "\n".join(lines)


def retry_prompt(last_error: str) -> str:
    return (
        f"Your previous response was invalid ({last_error}). "
        "Respond again with ONLY the JSON object described above."
    )
