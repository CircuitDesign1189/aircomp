# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Does conditioning the receiver on the decoded embedding actually change its
behavior, or does injecting it do nothing observable?

`reconstruction.py`'s embed-check shows the decoded embedding stays
cosine-close to the true target through the channel -- a property of the
vector, not of what the LLM does with it. This module asks the behavioral
question directly: does the receiving LLM's own log-likelihood of the true
content shift toward "more likely" when conditioned on that example's own
decoded embedding, versus a real embedding decoded for a DIFFERENT example?

This is a paired discrimination test (chance = 50% win rate), not "does the
model spontaneously generate the right JSON" -- nothing in this pipeline
trains the LLM itself to interpret an injected vector, so the sensitive
question is whether conditioning shifts probability mass at all, not whether
it produces fluent correct text unaided. See `LocalLLM.score_completion_with_soft_prompt`.
"""
from __future__ import annotations

import random

import torch

from airComp.channel.analog import AnalogAWGNChannel
from airComp.config import ITEM_TYPES
from airComp.jscc.dataset import IDX_TO_ACTION, offer_canonical_text
from airComp.jscc.modules import pool_to_mask

#: Minimum win rate above chance (0.5) to call this a real effect. Loose, like
#: MIN_INPUT_DEPENDENCE in reconstruction.py -- a smoke test for "any
#: measurable effect at all", not a quality bar.
MIN_WIN_RATE_ABOVE_CHANCE = 0.10

RECEIVER_SYSTEM_PROMPT = (
    "You are one of two agents negotiating how to split a pool of items. You just "
    "received a message from the other agent, but it arrived over a channel you can "
    "only sense, not read as text."
)
RECEIVER_USER_PROMPT = "State what you believe the other agent just communicated."


def _decode_embed(encoder, decoder, hidden: torch.Tensor, mask: torch.Tensor, snr_db) -> torch.Tensor:
    with torch.no_grad():
        z = encoder(hidden.unsqueeze(0))
        y = z if snr_db is None else AnalogAWGNChannel()(z, float(snr_db))
        return decoder(y, mask.unsqueeze(0))["embed"][0]


def injection_discrimination(
    llm,
    encoder,
    decoder,
    examples,
    snr_grid=(None, 0.0, -10.0),
    n: int = 30,
    max_count: int = 4,
    seed: int = 0,
) -> dict:
    """Paired likelihood discrimination per SNR (including `None` = noiseless).

    For each of `n` sampled examples: decode its own embedding and a randomly
    paired different example's embedding (both through the same channel),
    score the model's log-likelihood of the example's own true text under
    each, plus a zero-embedding floor, and count how often "own" wins.
    """
    rng = random.Random(seed)
    sample = examples if len(examples) <= n else rng.sample(examples, n)

    results: dict = {}
    for snr_db in snr_grid:
        label = "noiseless" if snr_db is None else f"snr_{snr_db:+.0f}"
        own_scores, mismatched_scores, zero_scores, wins = [], [], [], 0

        for ex in sample:
            other = rng.choice([e for e in examples if e is not ex])
            own_embed = _decode_embed(encoder, decoder, ex.hidden.float(),
                                      pool_to_mask(ex.pool, ITEM_TYPES, max_count), snr_db)
            mismatched_embed = _decode_embed(encoder, decoder, other.hidden.float(),
                                             pool_to_mask(other.pool, ITEM_TYPES, max_count), snr_db)
            zero_embed = torch.zeros_like(own_embed)

            true_text = offer_canonical_text(IDX_TO_ACTION[ex.action_idx], ex.counts)
            own_ll = llm.score_completion_with_soft_prompt(
                RECEIVER_SYSTEM_PROMPT, [], RECEIVER_USER_PROMPT, own_embed, true_text)
            mismatched_ll = llm.score_completion_with_soft_prompt(
                RECEIVER_SYSTEM_PROMPT, [], RECEIVER_USER_PROMPT, mismatched_embed, true_text)
            zero_ll = llm.score_completion_with_soft_prompt(
                RECEIVER_SYSTEM_PROMPT, [], RECEIVER_USER_PROMPT, zero_embed, true_text)

            own_scores.append(own_ll)
            mismatched_scores.append(mismatched_ll)
            zero_scores.append(zero_ll)
            wins += int(own_ll > mismatched_ll)

        n_scored = len(sample)
        results[label] = {
            "n": n_scored,
            "win_rate": wins / n_scored,
            "own_ll_mean": sum(own_scores) / n_scored,
            "mismatched_ll_mean": sum(mismatched_scores) / n_scored,
            "zero_ll_mean": sum(zero_scores) / n_scored,
        }
    return results


def verdict(results: dict) -> tuple[bool, str]:
    """(behavior changes?, one-line explanation)."""
    label, best = max(results.items(), key=lambda kv: kv[1]["win_rate"])
    gap = best["win_rate"] - 0.5
    if gap < MIN_WIN_RATE_ABOVE_CHANCE:
        return False, (
            f"best win rate {best['win_rate']:.2f} (at {label}) is not meaningfully above chance "
            f"(0.50 + {MIN_WIN_RATE_ABOVE_CHANCE}); the receiver's likelihood does not discriminate "
            f"which message it actually received."
        )
    return True, f"win rate {best['win_rate']:.2f} at {label}, meaningfully above chance (0.50)."


def format_injection_table(results: dict) -> str:
    lines = [f"{'condition':>12} {'n':>4} {'win_rate':>9} {'own_ll':>9} {'mismatch_ll':>12} {'zero_ll':>9}"]
    for label, r in results.items():
        lines.append(
            f"{label:>12} {r['n']:>4} {r['win_rate']:>9.2f} {r['own_ll_mean']:>9.3f} "
            f"{r['mismatched_ll_mean']:>12.3f} {r['zero_ll_mean']:>9.3f}"
        )
    ok, why = verdict(results)
    lines.append("")
    lines.append(("PASS: " if ok else "FAIL: ") + why)
    return "\n".join(lines)
