"""Unit tests for injection_discrimination's aggregation/verdict logic.

Uses stub encoder/decoder/LLM -- no real model needed. The question here is
"does the win-rate/verdict plumbing work", not "does the real LLM discriminate"
(that needs `pytest -m slow`; see tests/test_llm_soft_prompt.py).
"""
from __future__ import annotations

import torch

from airComp.config import ITEM_TYPES
from airComp.env.negotiation import Pool
from airComp.eval.injection_check import (
    MIN_WIN_RATE_ABOVE_CHANCE,
    format_injection_table,
    injection_discrimination,
    verdict,
)
from airComp.jscc.dataset import IDX_TO_ACTION, JsccExample, offer_canonical_text

MAX_COUNT = 2


class _IdentityEncoder:
    def __call__(self, hidden):
        return hidden


class _IdentityDecoder:
    def __call__(self, y, mask):
        return {"embed": y}


#: Deliberately distinct canonical texts -- offer_canonical_text drops zero
#: counts, so a modulo-generated sweep collides (e.g. book=0,hat=1,ball=2 and
#: its repeat both render the same string); these six do not.
_DISTINCT_COUNTS = [
    {"book": 0, "hat": 0, "ball": 0},
    {"book": 1, "hat": 0, "ball": 0},
    {"book": 0, "hat": 1, "ball": 0},
    {"book": 0, "hat": 0, "ball": 1},
    {"book": 2, "hat": 0, "ball": 0},
    {"book": 0, "hat": 2, "ball": 0},
]


def _examples(n: int = 6) -> list:
    pool = Pool(counts={t: MAX_COUNT for t in ITEM_TYPES})
    return [
        JsccExample(
            hidden=torch.tensor([float(i)]),
            action_idx=0,
            counts=_DISTINCT_COUNTS[i],
            aux=0.0,
            pool=pool,
            values={},
        )
        for i in range(n)
    ]


class _PerfectDiscriminatorLLM:
    """Scores 1.0 iff the injected embedding's identity matches the text's
    owning example, 0.0 otherwise -- the identity encoder/decoder pass hidden
    straight through unchanged, so this is exact and noise-free."""

    def __init__(self, examples):
        self.text_to_value = {
            offer_canonical_text(IDX_TO_ACTION[ex.action_idx], ex.counts): ex.hidden.item() for ex in examples
        }

    def score_completion_with_soft_prompt(self, system_prompt, history, user_prompt, soft_prompt_embed, completion_text):
        return 1.0 if soft_prompt_embed.item() == self.text_to_value[completion_text] else 0.0


class _ChanceLLM:
    """Ignores the embedding entirely -- own and mismatched always tie."""

    def score_completion_with_soft_prompt(self, system_prompt, history, user_prompt, soft_prompt_embed, completion_text):
        return 0.0


def test_perfect_discriminator_scores_full_win_rate():
    examples = _examples()
    results = injection_discrimination(
        _PerfectDiscriminatorLLM(examples), _IdentityEncoder(), _IdentityDecoder(), examples,
        snr_grid=[None], n=len(examples), max_count=MAX_COUNT,
    )

    assert results["noiseless"]["win_rate"] == 1.0
    ok, why = verdict(results)
    assert ok, why


def test_chance_level_llm_is_caught():
    examples = _examples()
    results = injection_discrimination(
        _ChanceLLM(), _IdentityEncoder(), _IdentityDecoder(), examples,
        snr_grid=[None], n=len(examples), max_count=MAX_COUNT,
    )

    # ties count as a loss (own_ll > mismatched_ll is False when equal)
    assert results["noiseless"]["win_rate"] == 0.0
    ok, why = verdict(results)
    assert not ok
    assert "chance" in why


def test_verdict_threshold_is_relative_to_chance():
    just_above = {"noiseless": {"win_rate": 0.5 + MIN_WIN_RATE_ABOVE_CHANCE + 0.01, "n": 10,
                                "own_ll_mean": 0.0, "mismatched_ll_mean": 0.0, "zero_ll_mean": 0.0}}
    just_below = {"noiseless": {"win_rate": 0.5 + MIN_WIN_RATE_ABOVE_CHANCE - 0.01, "n": 10,
                                "own_ll_mean": 0.0, "mismatched_ll_mean": 0.0, "zero_ll_mean": 0.0}}

    assert verdict(just_above)[0]
    assert not verdict(just_below)[0]


def test_format_injection_table_states_the_verdict():
    examples = _examples()
    results = injection_discrimination(
        _PerfectDiscriminatorLLM(examples), _IdentityEncoder(), _IdentityDecoder(), examples,
        snr_grid=[None], n=len(examples), max_count=MAX_COUNT,
    )

    text = format_injection_table(results)

    assert "PASS" in text
    assert "win_rate" in text
