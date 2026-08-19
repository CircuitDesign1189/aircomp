"""Integration tests requiring the actual local LLM to be downloaded.
Run with `pytest -m slow -q`.
"""
from __future__ import annotations

import torch

import pytest

from airComp.agents.llm_backend import LocalLLM

SYSTEM = "You are a helpful assistant."
USER = "Say hello."


@pytest.mark.slow
def test_score_completion_with_soft_prompt_returns_a_finite_log_likelihood():
    llm = LocalLLM(device="cpu")
    embed = llm.embed_text("propose: book=2, hat=1, ball=0")

    ll = llm.score_completion_with_soft_prompt(SYSTEM, [], USER, embed, "propose: book=2, hat=1, ball=0")

    assert isinstance(ll, float)
    assert torch.isfinite(torch.tensor(ll))
    assert ll < 0  # a mean log-likelihood per token; never positive


@pytest.mark.slow
def test_soft_prompt_changes_the_score_relative_to_a_different_embedding():
    llm = LocalLLM(device="cpu")
    own_text = "propose: book=2, hat=1, ball=0"
    other_text = "reject"
    own_embed = llm.embed_text(own_text)
    other_embed = llm.embed_text(other_text)

    own_scores_own_embed = llm.score_completion_with_soft_prompt(SYSTEM, [], USER, own_embed, own_text)
    own_scores_other_embed = llm.score_completion_with_soft_prompt(SYSTEM, [], USER, other_embed, own_text)

    # Not asserting which direction wins (that's the actual research question,
    # answered by evaluate.py injection-check) -- only that the embedding is
    # not being ignored by the forward pass.
    assert own_scores_own_embed != own_scores_other_embed
