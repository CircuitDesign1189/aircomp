"""Semantic/JSCC pipeline: LLM hidden state -> SemanticEncoder -> AnalogAWGNChannel
-> SemanticDecoder -> structured Offer.

The sender's hidden state is pooled only over the tokens generated for the
offer JSON (never private reasoning), matching what the baseline pipeline
would have serialized to text -- see CLAUDE.md's fairness note. The decoded
offer feeds back into `history_prompt` on the next turn exactly like a
baseline-pipeline offer would, so the LLM's own reasoning stays text-based
and pipeline-agnostic; only the turn's communicative payload crosses the
channel differently.
"""
from __future__ import annotations

from typing import Optional

import torch

from airComp.agents.llm_backend import LocalLLM
from airComp.agents.prompts import history_prompt, system_prompt
from airComp.channel.analog import AnalogAWGNChannel
from airComp.config import ITEM_TYPES
from airComp.env.negotiation import Offer, Pool, TurnRecord, Values
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder, pool_to_mask

ACTIONS = ("propose", "accept", "reject")


class SemanticAgent:
    def __init__(
        self,
        llm: LocalLLM,
        encoder: SemanticEncoder,
        decoder: SemanticDecoder,
        snr_db: float,
        max_messages: int = 10,
        max_count: int = 4,
        include_rationale: bool = True,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        device: str = "cpu",
    ):
        self.llm = llm
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.channel = AnalogAWGNChannel()
        self.snr_db = snr_db
        self.max_messages = max_messages
        self.max_count = max_count
        self.include_rationale = include_rationale
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def take_turn(
        self,
        pool: Pool,
        own_values: Values,
        round_index: int,
        history: list,
        standing_offer: Optional[Offer],
    ) -> TurnRecord:
        sys_prompt = system_prompt(pool, own_values, self.max_messages, self.include_rationale)
        user_prompt = history_prompt(history, standing_offer)

        _text, hidden = self.llm.chat_with_hidden(
            sys_prompt, [], user_prompt, self.max_new_tokens, self.temperature
        )

        hidden = hidden.to(self.device).unsqueeze(0)
        z = self.encoder(hidden)
        y = self.channel(z, self.snr_db)
        mask = pool_to_mask(pool, ITEM_TYPES, self.max_count).unsqueeze(0).to(self.device)
        out = self.decoder(y, mask)

        action_idx = int(torch.argmax(out["action_logits"], dim=-1).item())
        action = ACTIONS[action_idx]
        counts = None
        if action == "propose":
            count_idx = torch.argmax(out["offer_logits"], dim=-1)[0]  # (num_types,)
            counts = {t: int(count_idx[i].item()) for i, t in enumerate(ITEM_TYPES)}

        received_offer = Offer(action=action, counts=counts, message=None)
        channel_stats = {
            "snr_db": self.snr_db,
            "k": self.encoder.k,
            "aux": out["aux"][0].tolist(),
        }
        # A clean pre-channel discrete "raw offer" isn't recoverable from a continuous
        # latent without the decoder, so the post-decode offer is reported as both raw
        # and received, for TurnRecord shape parity with TextAgent.
        return TurnRecord(
            raw_offer=received_offer,
            received_offer=received_offer,
            parse_failed=False,
            channel_stats=channel_stats,
        )
