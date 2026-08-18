"""Preamble-correlation frame detection.

Index bookkeeping (this is where sign errors hide, so it is spelled out):

    taps        RRC, odd length L, group delay D = (L-1)/2
    tx          shape(symbols) -> symbol n peaks at n*sps + D
    y           matched_filter(rx) -> symbol n peaks at n*sps + 2D + d
                (d = unknown channel/capture delay)
    ref         shape of the preamble through the FULL raised cosine
                rc = taps (*) taps, so preamble symbol 0 peaks at index 2D
    corr[j]     = sum_i y[j+i] * conj(ref[i]), peaks at j = p*sps + d
                where p = layout.preamble_start

    => burst symbol 0 sits at  sym0 = j_peak - p*sps + 2D
       and symbol n is sampled at sym0 + n*sps.

Hardware triggering is deliberately not used. A clock-locked HackRF pair has no
frequency offset, so the capture delay d is unknown but *constant* within a
burst -- correlation alone is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps_signal

from hwlab.dsp import pulse


@dataclass(frozen=True)
class SyncResult:
    sym0_index: int
    peak_ratio: float  # |corr| peak / median(|corr|); unit-free detection confidence
    peak_index: int


def build_reference(preamble: np.ndarray, sps: int, taps: np.ndarray) -> np.ndarray:
    """Preamble shaped by the round-trip (TX RRC then RX matched filter)."""
    rc = np.convolve(taps, taps)
    return sps_signal.fftconvolve(pulse.upsample(preamble, sps), rc.astype(complex), mode="full")


def correlate(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """corr[j] = sum_i y[j+i] * conj(ref[i]) -- scipy conjugates `in2` for complex input."""
    if len(y) < len(ref):
        return np.zeros(0, dtype=complex)
    return sps_signal.correlate(y, ref, mode="valid", method="fft")


def find_burst(
    y: np.ndarray,
    ref: np.ndarray,
    preamble_start_sym: int,
    sps: int,
    taps: np.ndarray,
    total_symbols: int,
    min_peak_ratio: float = 4.0,
) -> SyncResult | None:
    """Locate a burst that fits entirely inside `y`.

    In repeat-transmit mode the capture usually contains several bursts and the
    first one is often truncated, so candidates are tried strongest-first and
    the first fully-contained one wins.
    """
    corr = correlate(y, ref)
    if corr.size == 0:
        return None
    mag = np.abs(corr)
    baseline = float(np.median(mag))
    if baseline <= 0:
        baseline = float(np.mean(mag)) or 1e-12

    two_d = len(taps) - 1
    span = (total_symbols - 1) * sps

    # Look only at a bounded number of candidates: correlation peaks are sparse.
    order = np.argsort(mag)[::-1][:64]
    for j in order:
        ratio = float(mag[j] / baseline)
        if ratio < min_peak_ratio:
            break
        sym0 = int(j) - preamble_start_sym * sps + two_d
        if sym0 >= 0 and sym0 + span < len(y):
            return SyncResult(sym0_index=sym0, peak_ratio=ratio, peak_index=int(j))
    return None


def sample_symbols(y: np.ndarray, sym0_index: int, sps: int, start_sym: int, count: int) -> np.ndarray:
    idx = sym0_index + (start_sym + np.arange(count)) * sps
    return y[idx]
