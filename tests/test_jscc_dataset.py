from __future__ import annotations

import torch

from airComp.config import NegotiationConfig
from airComp.jscc.dataset import collect_dataset, offer_canonical_text

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
