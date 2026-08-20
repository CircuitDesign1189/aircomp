# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from airComp.config import ITEM_TYPES
from airComp.config import NegotiationConfig
from airComp.env.negotiation import Pool
from airComp.jscc.dataset import JsccExample, backfill_embed_targets, collect_dataset, offer_canonical_text

VALID_REPLY = '{"action": "propose", "counts": {"book": 1, "hat": 0, "ball": 0}, "message": ""}'


class StubLLMNoEmbed:
    """Minimal stand-in for a backend with no embed_text -- the ONNX genai path."""

    def __init__(self, hidden_dim: int = 8):
        self.hidden_dim = hidden_dim
        self.calls = 0

    def chat_with_hidden(self, system_prompt, history, user_prompt):
        self.calls += 1
        return VALID_REPLY, torch.full((self.hidden_dim,), float(self.calls))


class StubLLMWithEmbed(StubLLMNoEmbed):
    """Adds embed_text, mirroring the CPU torch backend."""

    def embed_text(self, text: str) -> torch.Tensor:
        return torch.full((self.hidden_dim,), float(len(text)))


def test_offer_canonical_text_renders_propose_and_other_actions():
    assert offer_canonical_text("propose", {"book": 2, "hat": 0, "ball": 1}) == "propose: ball=1, book=2"
    assert offer_canonical_text("propose", {"book": 0, "hat": 0, "ball": 0}) == "propose: nothing"
    assert offer_canonical_text("accept", {}) == "accept"
    assert offer_canonical_text("reject", {}) == "reject"


def test_collect_dataset_leaves_embed_target_none_without_embed_text():
    cfg = NegotiationConfig(max_messages=2)
    examples = collect_dataset(StubLLMNoEmbed(), n_episodes=1, cfg=cfg)

    assert examples
    assert all(ex.embed_target is None for ex in examples)


def test_collect_dataset_fills_embed_target_when_backend_supports_it():
    cfg = NegotiationConfig(max_messages=2)
    examples = collect_dataset(StubLLMWithEmbed(), n_episodes=1, cfg=cfg)

    assert examples
    for ex in examples:
        expected_text = offer_canonical_text(
            "propose" if ex.action_idx == 0 else ("accept" if ex.action_idx == 1 else "reject"), ex.counts
        )
        assert ex.embed_target is not None
        assert torch.equal(ex.embed_target, torch.full((8,), float(len(expected_text))))


def test_backfill_embed_targets_needs_no_new_episodes():
    """The key cost claim: backfilling is a pure function of (action, counts),
    not a re-collection -- the stub here has no chat_with_hidden at all."""
    pool = Pool(counts={t: 4 for t in ITEM_TYPES})
    examples = [
        JsccExample(hidden=torch.zeros(4), action_idx=0, counts={"book": 2, "hat": 0, "ball": 0},
                    aux=0.0, pool=pool, values={}),
        JsccExample(hidden=torch.zeros(4), action_idx=1, counts={"book": 0, "hat": 0, "ball": 0},
                    aux=1.0, pool=pool, values={}),
    ]
    assert all(ex.embed_target is None for ex in examples)

    backfill_embed_targets(examples, StubLLMWithEmbed())

    assert torch.equal(examples[0].embed_target, torch.full((8,), float(len("propose: book=2"))))
    assert torch.equal(examples[1].embed_target, torch.full((8,), float(len("accept"))))
