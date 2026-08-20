# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Least-squares channel estimation and zero-forcing equalization.

The link is narrowband (100 ksym/s) and, in the conducted configuration, has no
multipath at all, so the channel is a single complex gain h that lumps together
the attenuator, the TX/RX gain settings, cable phase, and the arbitrary phase
left over from the receiver's capture offset.

h is estimated from the PREAMBLE rather than the pilots -- see the note in
`hwlab/dsp/burst.py` -- which leaves the pilot residual as a genuinely
independent measure of the noise.

CRITICAL: after dividing by h, do NOT renormalize z_hat to ||z_hat|| = sqrt(k).
`SemanticDecoder` was trained on "unit-power signal plus noise" inputs. Forcing
the norm back to sqrt(k) rescales the noise along with the signal and destroys
the very SNR relationship the experiment is measuring. Dividing by the
estimated gain already restores the correct absolute scale.
"""
from __future__ import annotations

import numpy as np


def estimate_gain(rx_pilots: np.ndarray, tx_pilots: np.ndarray) -> complex:
    """LS estimate of the flat complex channel gain."""
    denom = np.sum(np.abs(tx_pilots) ** 2)
    if denom <= 0:
        raise ValueError("pilot sequence has zero energy")
    return complex(np.sum(rx_pilots * np.conj(tx_pilots)) / denom)


def equalize(symbols: np.ndarray, h: complex) -> np.ndarray:
    if h == 0:
        raise ValueError("channel estimate is exactly zero -- no signal captured")
    return symbols / h


def pilot_residual_noise_var_per_real(rx_pilots: np.ndarray, tx_pilots: np.ndarray, h: complex) -> float:
    """Noise variance per real component, measured on the equalized pilots.

    Independent cross-check on the guard-region estimate in `measure.py`. Noisy
    (only `n_pilots` samples) but it is measured exactly where the data sits, so
    a large disagreement between the two points at a real problem -- typically a
    spur landing in-band, or a burst that was misaligned by one sample.
    """
    residual = equalize(rx_pilots, h) - tx_pilots
    return float(np.mean(np.abs(residual) ** 2) / 2.0)


def iq_imbalance_metrics(rx_pilots: np.ndarray, tx_pilots: np.ndarray) -> dict:
    """Rough image-rejection check.

    A zero-IF receiver with IQ gain/phase imbalance leaks a conjugate image. If
    the conjugate correlation is not far below the direct one, revisit the IF
    offset and the DC-offset correction before trusting any SNR numbers.
    """
    direct = np.abs(np.sum(rx_pilots * np.conj(tx_pilots)))
    image = np.abs(np.sum(rx_pilots * tx_pilots))
    ratio_db = 20.0 * np.log10(direct / image) if image > 0 else float("inf")
    return {"direct": float(direct), "image": float(image), "image_rejection_db": float(ratio_db)}
