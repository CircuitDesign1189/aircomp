# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Pulse shaping, digital IF up/down-conversion, and 8-bit sample formatting.

Sample-domain convention used across the SDRBackend interface: IQ arrays are
complex float in *normalized full-scale units*, where +-1.0 maps to +-127 LSB
of the HackRF's 8-bit converters. Backends do the int8 conversion internally so
LoopbackBackend can emulate quantization identically to the real device.

The RRC taps are normalized to UNIT ENERGY. That is load-bearing: with a
unit-energy matched filter, white noise of per-real variance v at the sample
input produces noise of per-real variance v at the symbol output, so the noise
measured in the burst's guard region *is* the noise on the data symbols. See
`hwlab/dsp/measure.py`.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps_signal


def rrc_taps(sps: int, span_symbols: int, rolloff: float) -> np.ndarray:
    """Root-raised-cosine taps, odd length (integer group delay), unit energy."""
    n = span_symbols * sps
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - (n - 1) / 2.0) / float(sps)  # time in symbol periods
    b = float(rolloff)
    h = np.zeros(n, dtype=float)

    at_zero = np.isclose(t, 0.0)
    h[at_zero] = 1.0 + b * (4.0 / np.pi - 1.0)

    if b > 0:
        at_sing = np.isclose(np.abs(t), 1.0 / (4.0 * b))
        h[at_sing] = (b / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * b))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * b))
        )
    else:
        at_sing = np.zeros(n, dtype=bool)

    rest = ~(at_zero | at_sing)
    tr = t[rest]
    num = np.sin(np.pi * tr * (1.0 - b)) + 4.0 * b * tr * np.cos(np.pi * tr * (1.0 + b))
    den = np.pi * tr * (1.0 - (4.0 * b * tr) ** 2)
    h[rest] = num / den

    return h / np.sqrt(np.sum(h**2))


def group_delay(taps: np.ndarray) -> int:
    return (len(taps) - 1) // 2


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    out = np.zeros(len(symbols) * sps, dtype=complex)
    out[::sps] = symbols
    return out


def shape(symbols: np.ndarray, sps: int, taps: np.ndarray) -> np.ndarray:
    """Zero-stuff + RRC. Symbol n peaks at index n*sps + group_delay(taps)."""
    return sps_signal.fftconvolve(upsample(symbols, sps), taps.astype(complex), mode="full")


def mix_up(x: np.ndarray, fs: float, if_offset_hz: float, start_index: int = 0) -> np.ndarray:
    n = np.arange(start_index, start_index + len(x))
    return x * np.exp(2j * np.pi * if_offset_hz * n / fs)


def mix_down(x: np.ndarray, fs: float, if_offset_hz: float, start_index: int = 0) -> np.ndarray:
    """Inverse of mix_up.

    The receiver's `start_index` is generally NOT the transmitter's, because the
    capture window starts at an arbitrary sample. That leaves a constant phase
    offset, which the pilot-based channel estimate absorbs. It stays *constant*
    only because the two HackRFs share a 10 MHz reference (CLKOUT -> CLKIN); an
    unlocked pair would leave a drifting phase that 8 data symbols cannot track.
    """
    n = np.arange(start_index, start_index + len(x))
    return x * np.exp(-2j * np.pi * if_offset_hz * n / fs)


def matched_filter(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """RRC is real and symmetric, so convolution is the matched filter."""
    return sps_signal.fftconvolve(x, taps.astype(complex), mode="full")


def quantize_int8(iq: np.ndarray) -> np.ndarray:
    """Full-scale float complex -> interleaved int8 (hackrf_transfer wire format)."""
    scaled = np.empty(2 * len(iq), dtype=float)
    scaled[0::2] = iq.real * 127.0
    scaled[1::2] = iq.imag * 127.0
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)


def dequantize_int8(buf: np.ndarray) -> np.ndarray:
    """Interleaved int8 -> full-scale float complex."""
    buf = np.asarray(buf, dtype=np.int8)
    if len(buf) % 2 != 0:
        buf = buf[:-1]
    return (buf[0::2].astype(float) + 1j * buf[1::2].astype(float)) / 127.0


def clipping_fraction(iq: np.ndarray) -> float:
    """Fraction of real components that would clip at full scale."""
    comps = np.concatenate([iq.real, iq.imag])
    return float(np.mean(np.abs(comps) >= 1.0))
