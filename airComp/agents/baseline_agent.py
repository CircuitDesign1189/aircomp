# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Conventional pipeline: LLM generates JSON text -> DigitalChannel -> parsed Offer.

Local formatting retries (bounded, `max_retries`) fix the agent's own
malformed output *before* anything is transmitted. The channel is a separate,
independent failure mode applied to the agent's own best (already-valid) text
-- this is what makes the SNR sweep comparison to the semantic pipeline fair.
"""
from __future__ import annotations

from typing import Optional

from airComp.agents.llm_backend import LocalLLM
from airComp.agents.parser import parse_offer, parse_offer_with_retries
from airComp.agents.prompts import history_prompt, retry_prompt, system_prompt
from airComp.channel.digital import DigitalChannel
from airComp.env.negotiation import Offer, Pool, TurnRecord, Values


class TextAgent:
    def __init__(
        self,
        llm: LocalLLM,
        channel: DigitalChannel,
        snr_db: float,
        max_messages: int = 10,
        max_retries: int = 2,
        include_rationale: bool = True,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.llm = llm
        self.channel = channel
        self.snr_db = snr_db
        self.max_messages = max_messages
        self.max_retries = max_retries
        self.include_rationale = include_rationale
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def take_turn(
        self,
        pool: Pool,
        own_values: Values,
        round_index: int,
        history: list,
        standing_offer: Optional[Offer],
    ) -> TurnRecord:
        sys_prompt = system_prompt(pool, own_values, self.max_messages, self.include_rationale)
        base_user_prompt = history_prompt(history, standing_offer)

        def generate_fn(attempt: int, last_error: Optional[str]) -> str:
            prompt = base_user_prompt if last_error is None else retry_prompt(last_error)
            return self.llm.chat(sys_prompt, [], prompt, self.max_new_tokens, self.temperature)

        raw_offer, raw_text, attempts = parse_offer_with_retries(generate_fn, pool, self.max_retries)

        if raw_offer is None:
            # Local generation never produced valid output, even after retries -- no point
            # sending known-garbage through the channel. This is a real, measured failure mode.
            return TurnRecord(
                raw_offer=None,
                received_offer=None,
                parse_failed=True,
                channel_stats={"attempts": attempts, "skipped_channel": True},
            )

        received_text, channel_stats = self.channel.transmit(raw_text, self.snr_db)
        channel_stats["attempts"] = attempts

        if received_text is None:
            return TurnRecord(raw_offer=raw_offer, received_offer=None, parse_failed=True, channel_stats=channel_stats)

        result = parse_offer(received_text, pool)
        return TurnRecord(
            raw_offer=raw_offer,
            received_offer=result.offer,
            parse_failed=not result.ok,
            channel_stats=channel_stats,
        )
