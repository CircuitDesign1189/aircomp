# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Fair digital baseline: LLM generates JSON -> source-coded to 8 bits -> channel.

`TextAgent` puts the LLM's entire completion on the wire (600-1350 bits of prose
and JSON per message) to convey an act worth 6.1 bits. Comparing that against a
16-symbol semantic latent measures source coding, not joint source-channel
coding. `CompactAgent` removes that confound: same prompts, same generation,
same bounded retries, same BPSK/AWGN channel -- only the payload changes, from
the completion text to `airComp/baseline/offer_codec`'s fixed 8-bit frame.

Two channel modes give the decomposition:
  "raw" -- 8 uncoded channel uses. Against TextAgent this isolates source coding.
  "fec" -- Hamming(7,4), 16 channel uses. That is exactly the semantic
           pipeline's k=16 real channel uses at the same SNR per real
           dimension, so against SemanticAgent it isolates the one thing this
           project actually claims: that an analog latent degrades better than
           bits do.

Known asymmetry, quantified by the +40 dB ceiling run rather than argued away:
CompactAgent must parse its own JSON in order to encode it, so it inherits
TextAgent's generation-failure mode. SemanticAgent discards the text and pools
the hidden state, so it has no such mode.
"""
from __future__ import annotations

from typing import Optional

from airComp.agents.parser import parse_offer_with_retries
from airComp.agents.prompts import history_prompt, retry_prompt, system_prompt
from airComp.baseline.offer_codec import OFFER_BITS, bits_to_offer, offer_to_bits
from airComp.channel.digital import DigitalChannel
from airComp.env.negotiation import Pool, TurnRecord, Values


class CompactAgent:
    def __init__(
        self,
        llm,
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
        standing_offer: Optional[object],
    ) -> TurnRecord:
        # Generation is deliberately identical to TextAgent's, down to the retry
        # prompts: with paired seeds the two pipelines then make the same LLM
        # calls, and any difference in outcome is the payload and nothing else.
        sys_prompt = system_prompt(pool, own_values, self.max_messages, self.include_rationale)
        base_user_prompt = history_prompt(history, standing_offer)

        def generate_fn(attempt: int, last_error: Optional[str]) -> str:
            prompt = base_user_prompt if last_error is None else retry_prompt(last_error)
            return self.llm.chat(sys_prompt, [], prompt, self.max_new_tokens, self.temperature)

        raw_offer, _raw_text, attempts = parse_offer_with_retries(generate_fn, pool, self.max_retries)

        if raw_offer is None:
            return TurnRecord(
                raw_offer=None,
                received_offer=None,
                parse_failed=True,
                channel_stats={"attempts": attempts, "skipped_channel": True},
            )

        bits = offer_to_bits(raw_offer, pool)
        received_bits, channel_stats = self.channel.transmit_bits(bits, self.snr_db)
        channel_stats["attempts"] = attempts
        channel_stats["payload_bits"] = OFFER_BITS

        received_offer = bits_to_offer(received_bits, pool)
        return TurnRecord(
            raw_offer=raw_offer,
            received_offer=received_offer,
            parse_failed=received_offer is None,
            channel_stats=channel_stats,
        )
