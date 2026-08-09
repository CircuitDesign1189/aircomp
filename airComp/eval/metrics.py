"""Per-run aggregate metrics over a list of EpisodeRecords."""
from __future__ import annotations

import numpy as np

from airComp.env.negotiation import EpisodeRecord
from airComp.env.scoring import pareto_efficiency as _pareto_efficiency
from airComp.env.scoring import social_welfare as _social_welfare


def agreement_rate(records: list) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.outcome == "agreement") / len(records)


def avg_utility(records: list, agent: str) -> float:
    if not records:
        return 0.0
    key = "utility_a" if agent == "A" else "utility_b"
    return float(np.mean([getattr(r, key) for r in records]))


def avg_social_welfare(records: list) -> float:
    if not records:
        return 0.0
    return float(np.mean([_social_welfare(r) for r in records]))


def avg_pareto_efficiency(records: list) -> float:
    agreements = [r for r in records if r.outcome == "agreement"]
    if not agreements:
        return 0.0
    return float(np.mean([_pareto_efficiency(r) for r in agreements]))


def rounds_to_agreement(records: list) -> float:
    agreements = [r for r in records if r.outcome == "agreement"]
    if not agreements:
        return float("nan")
    return float(np.mean([len(r.turns) for r in agreements]))


def effective_bits(record: EpisodeRecord, pipeline: str) -> float:
    """Total transmitted payload across the episode's turns.

    pipeline="digital": sums each turn's channel_stats["n_bits"].
    pipeline="semantic": sums each turn's channel_stats["k"] (raw symbol count).
    These are NOT directly comparable -- see `semantic_bits_equivalent` and the
    bandwidth-fairness caveat in CLAUDE.md/README.
    """
    key = "n_bits" if pipeline == "digital" else "k"
    return float(sum(t.channel_stats.get(key, 0) for t in record.turns))


def semantic_bits_equivalent(record: EpisodeRecord) -> float:
    """Shannon-capacity-equivalent bit estimate: k * 0.5*log2(1+SNR_linear) per turn, summed."""
    total = 0.0
    for t in record.turns:
        k = t.channel_stats.get("k")
        snr_db = t.channel_stats.get("snr_db")
        if k is None or snr_db is None:
            continue
        snr_linear = 10 ** (snr_db / 10.0)
        total += k * 0.5 * np.log2(1 + snr_linear)
    return float(total)
