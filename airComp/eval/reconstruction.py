"""Offer-reconstruction accuracy — does the decoder actually use the channel?

The task metrics (agreement rate, social welfare, Pareto efficiency) saturate.
Measured over the real radio, they were flat from -10 dB to +20 dB. Two very
different situations produce that same flat line:

  1. the pipeline is genuinely robust across the range, or
  2. the decoder has learned to ignore its input and emit a prior.

No task metric can tell those apart, because a decoder that always proposes the
modal split still reaches agreement in a forgiving bargaining game. The
discriminating measurement is how well the decoded offer matches the offer that
was actually sent, scored against two floors: the same decoder fed pure noise,
and a constant predictor that always emits the modal count.

Run this whenever the semantic curve looks suspiciously flat, and before making
any claim that the semantic pipeline "degrades gracefully" — graceful
degradation and no communication at all look identical from the task side.
"""
from __future__ import annotations

from collections import Counter

import torch

from airComp.channel.analog import AnalogAWGNChannel
from airComp.config import ITEM_TYPES
from airComp.jscc.modules import pool_to_mask

#: Minimum gap in exact-offer accuracy between real signal and pure noise for the
#: pipeline to count as communicating at all. Deliberately loose: this is a
#: smoke test for "the decoder is a constant", not a quality bar.
MIN_INPUT_DEPENDENCE = 0.10


def _decode(decoder, y: torch.Tensor, masks: torch.Tensor):
    out = decoder(y, masks)
    return out["offer_logits"].argmax(dim=-1), out["action_logits"].argmax(dim=-1)


def _score(counts, actions, truth, truth_actions) -> dict:
    return {
        "exact_offer": (counts == truth).all(dim=1).float().mean().item(),
        "per_item": (counts == truth).float().mean().item(),
        "action": (actions == truth_actions).float().mean().item(),
    }


def _tensors(examples, max_count: int):
    truth = torch.tensor([[e.counts[t] for t in ITEM_TYPES] for e in examples])
    actions = torch.tensor([e.action_idx for e in examples])
    masks = torch.stack([pool_to_mask(e.pool, ITEM_TYPES, max_count) for e in examples])
    hidden = torch.stack([e.hidden for e in examples]).float()
    return hidden, truth, actions, masks


def modal_constant(truth: torch.Tensor) -> dict:
    """What a decoder that ignores its input entirely would score."""
    modal = torch.tensor(
        [Counter(truth[:, i].tolist()).most_common(1)[0][0] for i in range(truth.shape[1])]
    )
    const = modal.unsqueeze(0).expand_as(truth)
    return {
        "exact_offer": (const == truth).all(dim=1).float().mean().item(),
        "per_item": (const == truth).float().mean().item(),
        "counts": modal.tolist(),
    }


def reconstruction_table(
    encoder,
    decoder,
    examples,
    snr_grid=(-10.0, -5.0, 0.0, 5.0, 10.0, 20.0),
    max_count: int = 4,
    seed: int = 0,
) -> dict:
    """Score the decoder on real signal, on pure noise, and against the modal constant.

    `examples` are JsccExample records (see airComp/jscc/dataset.py). Scoring on
    the training set is fine here and is what the caller usually has: the
    question is input-dependence, not generalization, and both signal and noise
    are scored on the same records.
    """
    torch.manual_seed(seed)
    was_training = encoder.training or decoder.training
    encoder.eval()
    decoder.eval()

    hidden, truth, actions, masks = _tensors(examples, max_count)
    channel = AnalogAWGNChannel()
    conditions: dict = {}

    with torch.no_grad():
        z = encoder(hidden)
        conditions["noiseless"] = _score(*_decode(decoder, z, masks), truth, actions)
        for snr in snr_grid:
            conditions[f"snr_{snr:+.0f}"] = _score(
                *_decode(decoder, channel(z, float(snr)), masks), truth, actions
            )
        pure = torch.randn_like(z) * z.pow(2).mean().sqrt()
        conditions["pure_noise"] = _score(*_decode(decoder, pure, masks), truth, actions)
        conditions["zeros"] = _score(
            *_decode(decoder, torch.zeros_like(z), masks), truth, actions
        )

    if was_training:
        encoder.train()
        decoder.train()

    best_signal = max(
        v["exact_offer"] for k, v in conditions.items() if k == "noiseless" or k.startswith("snr_")
    )
    floor = max(conditions["pure_noise"]["exact_offer"], conditions["zeros"]["exact_offer"])
    return {
        "n": len(examples),
        "conditions": conditions,
        "modal_constant": modal_constant(truth),
        "input_dependence": best_signal - floor,
    }


def verdict(table: dict) -> tuple[bool, str]:
    """(communicating?, one-line explanation) — the thing worth putting in a report."""
    gap = table["input_dependence"]
    modal = table["modal_constant"]["exact_offer"]
    best = max(
        v["exact_offer"]
        for k, v in table["conditions"].items()
        if k == "noiseless" or k.startswith("snr_")
    )
    if gap < MIN_INPUT_DEPENDENCE:
        return False, (
            f"decoder looks input-independent: best signal {best:.3f} vs noise/zero floor "
            f"{best - gap:.3f} (gap {gap:.3f} < {MIN_INPUT_DEPENDENCE}). Task metrics from this "
            f"checkpoint mean nothing."
        )
    if best <= modal:
        return False, (
            f"decoder does not beat the modal constant ({best:.3f} vs {modal:.3f}); it carries "
            f"no more information than the prior."
        )
    return True, (
        f"decoder uses the channel: best signal {best:.3f}, noise/zero floor {best - gap:.3f}, "
        f"modal constant {modal:.3f}."
    )


def format_table(table: dict) -> str:
    lines = [f"n = {table['n']}", f"{'condition':>14} {'exact':>7} {'per-item':>9} {'action':>7}"]
    for name, s in table["conditions"].items():
        lines.append(f"{name:>14} {s['exact_offer']:>7.3f} {s['per_item']:>9.3f} {s['action']:>7.3f}")
    m = table["modal_constant"]
    lines.append(f"{'modal const':>14} {m['exact_offer']:>7.3f} {m['per_item']:>9.3f} {'-':>7}"
                 f"   (counts {m['counts']})")
    ok, why = verdict(table)
    lines.append("")
    lines.append(("PASS: " if ok else "FAIL: ") + why)
    return "\n".join(lines)
