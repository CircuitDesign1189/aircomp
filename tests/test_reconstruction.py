"""The check that separates "robust" from "not communicating at all".

The hardware sweep produced a flat task-metric curve from -10 to +20 dB. That is
what genuine robustness looks like, and it is also exactly what a decoder that
ignores its input looks like. These tests pin the discriminator, because the
cost of getting it wrong is publishing a graceful-degradation claim about a
constant.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from airComp.config import ITEM_TYPES
from airComp.env.negotiation import Pool
from airComp.eval.reconstruction import (
    MIN_INPUT_DEPENDENCE,
    embed_reconstruction_table,
    embed_verdict,
    format_embed_table,
    format_table,
    reconstruction_table,
    verdict,
)
from airComp.jscc.losses import embed_cosine_loss
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder, pool_to_mask

K = 8
MAX_COUNT = 2
EMBED_DIM = 12
CLASSES = [(0, 0, 0), (1, 0, 1), (2, 1, 0), (0, 2, 2)]


def _examples(n: int = 128, input_dim: int = 16):
    """Trivially learnable: the hidden state is a noisy one-hot over 4 offer classes."""
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(n):
        cls = i % len(CLASSES)
        hidden = torch.zeros(input_dim)
        hidden[cls] = 1.0
        hidden += 0.01 * torch.randn(input_dim, generator=g)
        counts = dict(zip(ITEM_TYPES, CLASSES[cls]))
        out.append(
            SimpleNamespace(
                hidden=hidden,
                action_idx=cls % 3,
                counts=counts,
                pool=Pool(counts={t: MAX_COUNT for t in ITEM_TYPES}),
            )
        )
    return out


def _fit(examples, steps: int = 400):
    from airComp.jscc.modules import pool_to_mask

    encoder = SemanticEncoder(examples[0].hidden.numel(), (32, 32), K)
    decoder = SemanticDecoder(K, (32, 32), len(ITEM_TYPES), MAX_COUNT, 1)
    hidden = torch.stack([e.hidden for e in examples])
    truth = torch.tensor([[e.counts[t] for t in ITEM_TYPES] for e in examples])
    actions = torch.tensor([e.action_idx for e in examples])
    masks = torch.stack([pool_to_mask(e.pool, ITEM_TYPES, MAX_COUNT) for e in examples])

    opt = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)
    ce = torch.nn.CrossEntropyLoss()
    torch.manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        out = decoder(encoder(hidden), masks)
        loss = ce(out["offer_logits"].reshape(-1, MAX_COUNT + 1), truth.reshape(-1))
        loss = loss + ce(out["action_logits"], actions)
        loss.backward()
        opt.step()
    return encoder, decoder


def test_a_trained_decoder_is_reported_as_communicating():
    examples = _examples()
    encoder, decoder = _fit(examples)

    table = reconstruction_table(encoder, decoder, examples, (0.0, 20.0), MAX_COUNT)
    ok, why = verdict(table)

    assert ok, why
    assert table["input_dependence"] >= MIN_INPUT_DEPENDENCE


def test_a_constant_decoder_is_caught():
    """The failure this whole module exists for: output independent of the input."""
    examples = _examples()
    encoder, decoder = _fit(examples)
    with torch.no_grad():  # kill the input path; biases still produce a valid offer
        decoder.trunk[0].weight.zero_()
        decoder.trunk[0].bias.fill_(1.0)

    table = reconstruction_table(encoder, decoder, examples, (0.0, 20.0), MAX_COUNT)
    ok, why = verdict(table)

    assert not ok
    assert "input-independent" in why


def test_noise_and_zero_floors_are_measured():
    examples = _examples()
    encoder, decoder = _fit(examples)

    table = reconstruction_table(encoder, decoder, examples, (0.0,), MAX_COUNT)

    assert {"noiseless", "snr_+0", "pure_noise", "zeros"} <= set(table["conditions"])
    assert table["conditions"]["noiseless"]["exact_offer"] > table["conditions"]["pure_noise"]["exact_offer"]
    assert table["modal_constant"]["counts"] == [0, 0, 0]  # class (0,0,0) is modal-ish per type


def test_lower_snr_does_not_score_above_noiseless():
    """A sanity property of the sweep itself: noise must not help."""
    examples = _examples()
    encoder, decoder = _fit(examples)

    table = reconstruction_table(encoder, decoder, examples, (-15.0, 20.0), MAX_COUNT)

    c = table["conditions"]
    assert c["snr_-15"]["exact_offer"] <= c["noiseless"]["exact_offer"] + 1e-9


def test_format_table_states_the_verdict():
    examples = _examples()
    encoder, decoder = _fit(examples)

    text = format_table(reconstruction_table(encoder, decoder, examples, (0.0,), MAX_COUNT))

    assert "PASS" in text or "FAIL" in text
    assert "modal const" in text


def _examples_embed(n: int = 128, input_dim: int = 16):
    """Same trivially-learnable setup as `_examples`, but each class also gets
    a fixed random target embedding to stand in for `embed_target`."""
    g = torch.Generator().manual_seed(1)
    class_targets = torch.randn(len(CLASSES), EMBED_DIM, generator=g)
    out = []
    for i in range(n):
        cls = i % len(CLASSES)
        hidden = torch.zeros(input_dim)
        hidden[cls] = 1.0
        hidden += 0.01 * torch.randn(input_dim, generator=g)
        counts = dict(zip(ITEM_TYPES, CLASSES[cls]))
        out.append(
            SimpleNamespace(
                hidden=hidden,
                action_idx=cls % 3,
                counts=counts,
                pool=Pool(counts={t: MAX_COUNT for t in ITEM_TYPES}),
                embed_target=class_targets[cls].clone(),
            )
        )
    return out


def _fit_embed(examples, steps: int = 400):
    encoder = SemanticEncoder(examples[0].hidden.numel(), (32, 32), K)
    decoder = SemanticDecoder(K, (32, 32), len(ITEM_TYPES), MAX_COUNT, 1, embed_dim=EMBED_DIM)
    hidden = torch.stack([e.hidden for e in examples])
    target = torch.stack([e.embed_target for e in examples])
    masks = torch.stack([pool_to_mask(e.pool, ITEM_TYPES, MAX_COUNT) for e in examples])

    opt = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)
    torch.manual_seed(0)
    for _ in range(steps):
        opt.zero_grad()
        loss = embed_cosine_loss(decoder(encoder(hidden), masks)["embed"], target)
        loss.backward()
        opt.step()
    return encoder, decoder


def test_a_trained_embed_head_is_reported_as_communicating():
    examples = _examples_embed()
    encoder, decoder = _fit_embed(examples)

    table = embed_reconstruction_table(encoder, decoder, examples, (0.0, 20.0), MAX_COUNT)
    ok, why = embed_verdict(table)

    assert ok, why
    assert table["input_dependence"] >= MIN_INPUT_DEPENDENCE


def test_a_constant_embed_head_is_caught():
    """The embed-head analog of the failure reconstruction_table exists to catch."""
    examples = _examples_embed()
    encoder, decoder = _fit_embed(examples)
    with torch.no_grad():
        decoder.trunk[0].weight.zero_()
        decoder.trunk[0].bias.fill_(1.0)

    table = embed_reconstruction_table(encoder, decoder, examples, (0.0, 20.0), MAX_COUNT)
    ok, why = embed_verdict(table)

    assert not ok
    assert "input-independent" in why


def test_format_embed_table_states_the_verdict():
    examples = _examples_embed()
    encoder, decoder = _fit_embed(examples)

    text = format_embed_table(embed_reconstruction_table(encoder, decoder, examples, (0.0,), MAX_COUNT))

    assert "PASS" in text or "FAIL" in text
    assert "constant mean" in text


def test_eval_mode_is_restored():
    """The caller may hand in modules mid-training; don't silently leave them in eval."""
    examples = _examples(n=8)
    encoder, decoder = _fit(examples, steps=1)
    encoder.train()
    decoder.train()

    reconstruction_table(encoder, decoder, examples, (0.0,), MAX_COUNT)

    assert encoder.training and decoder.training
