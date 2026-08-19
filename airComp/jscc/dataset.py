"""Collect a supervised JSCC training set via frozen-LLM self-play over the
baseline (text) pipeline. Ground truth for each turn is whatever the LLM
itself intended to say that turn (its own parsed JSON) -- the decoder learns
to reproduce the sender's actual communicative intent from its hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from airComp.agents.llm_backend import LocalLLM
from airComp.agents.parser import parse_offer_with_retries
from airComp.agents.prompts import history_prompt, retry_prompt, system_prompt
from airComp.config import NegotiationConfig
from airComp.env.negotiation import Pool, TurnRecord, generate_pool, generate_values

ACTION_TO_IDX = {"propose": 0, "accept": 1, "reject": 2}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


@dataclass
class JsccExample:
    hidden: torch.Tensor  # (hidden_dim,)
    action_idx: int
    counts: dict  # item_type -> int; all zero when action != "propose"
    aux: float  # heuristic "concession rate" proxy in [0, 1]
    pool: Pool
    #: The acting agent's own per-unit values at this turn (item_type -> float).
    #: Needed for the Phase-2 expected-utility surrogate (airComp/jscc/losses.py);
    #: datasets collected before this field existed do not have it, so Phase-2
    #: training requires a fresh collect-dataset run.
    values: dict
    #: (hidden_dim,) mean-pooled input embedding of `offer_canonical_text(...)`,
    #: in the SAME model's own embedding space (LocalLLM.embed_text). Training
    #: target for SemanticDecoder's optional embed head -- see modules.py. None
    #: for datasets collected before this field existed, or via a backend that
    #: has no embed_text (only the CPU torch LocalLLM implements it).
    embed_target: torch.Tensor | None = None


def _concession_rate(round_index: int, max_messages: int) -> float:
    return round_index / max(max_messages - 1, 1)


def offer_canonical_text(action: str, counts: dict) -> str:
    """Deterministic text rendering of a turn's communicative content -- the
    string `LocalLLM.embed_text` embeds to produce `JsccExample.embed_target`
    (see `airComp/jscc/modules.py`'s injectable embed head). Terse on purpose:
    this is an embedding target, not a prompt meant to read naturally.
    """
    if action != "propose":
        return action
    items = ", ".join(f"{t}={n}" for t, n in sorted(counts.items()) if n > 0) or "nothing"
    return f"propose: {items}"


def collect_dataset(
    llm: LocalLLM,
    n_episodes: int,
    cfg: NegotiationConfig = NegotiationConfig(),
    seed_offset: int = 0,
) -> list:
    examples: list = []
    rng_master = np.random.default_rng(seed_offset)

    for _ in range(n_episodes):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        pool = generate_pool(rng, cfg)
        values_by_agent = {
            "A": generate_values(rng, pool, cfg),
            "B": generate_values(rng, pool, cfg),
        }
        first_mover = "A" if rng.integers(0, 2) == 0 else "B"
        order = ["A", "B"] if first_mover == "A" else ["B", "A"]

        history_turns: list = []
        standing_offer = None

        for turn_index in range(cfg.max_messages):
            active = order[turn_index % 2]
            own_values = values_by_agent[active]
            sys_prompt = system_prompt(pool, own_values, cfg.max_messages, cfg.include_rationale)
            user_prompt = history_prompt(history_turns, standing_offer)

            # Same bounded-retry path the baseline agent uses. Without it, a turn-0
            # formatting slip ends the episode with zero examples, and measurement on
            # this model put that at roughly half of all episodes -- so the collector
            # was throwing away half its runtime AND training the decoder on a
            # different distribution from the one it has to serve.
            hidden_by_attempt: list = []

            def generate_fn(attempt: int, last_error, sys_prompt=sys_prompt, user_prompt=user_prompt):
                prompt = user_prompt if last_error is None else retry_prompt(last_error)
                text, hidden = llm.chat_with_hidden(sys_prompt, [], prompt)
                hidden_by_attempt.append(hidden)
                return text

            offer, _, attempts = parse_offer_with_retries(generate_fn, pool, cfg.max_retries)
            if offer is None:
                break  # keep whatever examples this episode has already produced

            # Pool the hidden state of the attempt that actually parsed -- that is the
            # turn the receiver is being trained to reconstruct.
            hidden = hidden_by_attempt[attempts - 1]
            counts = offer.counts if offer.action == "propose" else {t: 0 for t in pool.counts}

            embed_target = None
            embed_fn = getattr(llm, "embed_text", None)
            if embed_fn is not None:
                embed_target = embed_fn(offer_canonical_text(offer.action, counts))

            examples.append(
                JsccExample(
                    hidden=hidden,
                    action_idx=ACTION_TO_IDX[offer.action],
                    counts=counts,
                    aux=_concession_rate(turn_index, cfg.max_messages),
                    pool=pool,
                    values=dict(own_values.per_unit),
                    embed_target=embed_target,
                )
            )

            history_turns.append(
                TurnRecord(turn_index=turn_index, agent=active, raw_offer=offer, received_offer=offer)
            )

            if offer.action in ("accept", "reject"):
                break
            standing_offer = offer

    return examples


def backfill_embed_targets(examples: list, llm) -> list:
    """Fill in `embed_target` on examples collected before that field existed
    (or via a backend without `embed_text`), in place -- no re-collection.

    `embed_target` is a pure function of `(action, counts)`; it never depended
    on which backend produced `hidden`, so this only needs `llm.embed_text`
    (an embedding-matrix lookup) run once per example, not a new negotiation
    episode or generation call.
    """
    for ex in examples:
        action = IDX_TO_ACTION[ex.action_idx]
        ex.embed_target = llm.embed_text(offer_canonical_text(action, ex.counts))
    return examples


def save_dataset(examples: list, path: str) -> None:
    torch.save(examples, path)


def load_dataset(path: str) -> list:
    return torch.load(path, weights_only=False)
