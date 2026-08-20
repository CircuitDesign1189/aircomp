# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Floor/ceiling normalisation and effective-SNR-gain, from sweep result files.

Why raw agreement rates cannot be compared directly
---------------------------------------------------
The pipelines do not share a zero or a one.

*Floor* -- what a pipeline scores when the channel carries no information at
all (measured at -60 dB). `SemanticDecoder` always emits a structurally valid
offer, so two agents drawing from the same prior agree 48% of the time by
coincidence. A digital frame that fails to decode is an implicit REJECT, so the
compact pipelines floor near zero. Reading 0.70 (semantic) against 0.06
(compact_fec) as a 0.64 advantage credits the semantic pipeline with its prior.

*Ceiling* -- what a pipeline scores when the channel is essentially lossless
(measured at +40 dB). This is below 1.0 for every pipeline because the LLM
sometimes fails to produce a usable offer, and it differs between pipelines
because `CompactAgent` must parse its own JSON to encode it while
`SemanticAgent` pools the hidden state and never parses at all.

So the comparable quantity is (p - floor) / (ceiling - floor): the fraction of
the achievable range the channel actually delivers. Effective SNR gain is then
the horizontal distance between two normalised curves at a matched level.
"""
from __future__ import annotations

import json
from typing import Optional

#: Sweep files disagree on how SNR keys are spelled ("-10" vs "-10.0") because
#: they were written by runs that passed the grid differently. Compare as floats.
def _by_snr(series: dict) -> dict:
    return {float(k): v for k, v in series.items()}


def load_curve(path: str, pipeline: str, metric: str = "agreement_rate") -> dict:
    """{snr: value} for one pipeline in one results file."""
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if pipeline not in results:
        raise KeyError(f"{pipeline!r} not in {path} (has {sorted(results)})")
    return {snr: summary[metric] for snr, summary in _by_snr(results[pipeline]).items()}


def normalize(curve: dict, floor: float, ceiling: float) -> dict:
    """Map a curve onto (p - floor) / (ceiling - floor)."""
    if ceiling <= floor:
        raise ValueError(f"ceiling {ceiling} must exceed floor {floor}")
    span = ceiling - floor
    return {snr: (p - floor) / span for snr, p in curve.items()}


def crossing(curve: dict, level: float) -> Optional[float]:
    """SNR at which the curve first rises through `level`, linearly interpolated.

    None if it never does -- which is itself a result, and must not be silently
    reported as an infinite gain.
    """
    points = sorted(curve.items())
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < level <= y1:
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return None


def effective_snr_gain(reference: dict, candidate: dict, level: float) -> Optional[float]:
    """dB by which `candidate` reaches `level` at a lower SNR than `reference`.

    Both curves must already be normalised, or the comparison silently mixes
    pipelines with different floors.
    """
    ref = crossing(reference, level)
    cand = crossing(candidate, level)
    if ref is None or cand is None:
        return None
    return ref - cand


def gain_table(reference: dict, candidate: dict, levels=(0.5, 0.9)) -> dict:
    """{level: {"reference_snr", "candidate_snr", "gain_db"}} for reporting."""
    out = {}
    for level in levels:
        ref, cand = crossing(reference, level), crossing(candidate, level)
        out[level] = {
            "reference_snr": ref,
            "candidate_snr": cand,
            "gain_db": None if ref is None or cand is None else ref - cand,
        }
    return out
