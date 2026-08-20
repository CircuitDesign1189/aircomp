# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Per-burst noise and SNR measurement.

The noise floor is measured in the burst's transmitted-zero guard region, in
the *matched-filter output* domain -- not on the raw samples. That distinction
matters: the receiver's analog baseband filter is wider than the signal, so raw
samples carry out-of-band noise the matched filter rejects. Measuring after the
matched filter, with unit-energy taps, gives exactly the noise variance that
lands on the data symbols.

Because the SNR reported here is measured rather than commanded, it is what
should label the x-axis of every hardware plot.
"""
from __future__ import annotations

import numpy as np

from hwlab.dsp.mapping import SIGNAL_POWER_PER_REAL, snr_db_from_powers


def noise_var_per_real_from_guard(y: np.ndarray, sym0_index: int, sps: int, guard: slice) -> float:
    """Variance per real component, from the matched-filtered guard region."""
    start = sym0_index + guard.start * sps
    stop = sym0_index + guard.stop * sps
    start = max(start, 0)
    stop = min(stop, len(y))
    seg = y[start:stop]
    if seg.size < 64:
        raise ValueError("guard region too short for a noise estimate")
    return float(np.mean(np.abs(seg) ** 2) / 2.0)


def snr_db_from_equalized_residual(residual_var_per_real: float) -> float:
    """SNR from a noise variance that is ALREADY in the post-equalization domain.

    Exists to stop callers dividing by |h|^2 a second time -- the pilot residual
    is measured after equalization, the guard estimate is measured before it.
    """
    return snr_db_from_powers(SIGNAL_POWER_PER_REAL, residual_var_per_real)


def measured_snr_db(h: complex, noise_var_per_real_: float) -> float:
    """SNR after zero-forcing equalization.

    Equalization scales the noise by 1/|h|^2 while restoring the signal to unit
    power per real component, so SNR = |h|^2 / noise_var_per_real.
    """
    if noise_var_per_real_ <= 0:
        return float("inf")
    equalized_noise = noise_var_per_real_ / (abs(h) ** 2)
    return snr_db_from_powers(SIGNAL_POWER_PER_REAL, equalized_noise)


#: An 8-bit converter's quantization noise is 1/sqrt(12) = 0.289 LSB rms. Below
#: this total rms it is within ~10 dB of everything else and starts to matter.
_DITHER_FLOOR_LSB = 1.0


def level_report(rx_iq: np.ndarray) -> dict:
    """ADC headroom check. Getting this wrong is the #1 time sink on real hardware."""
    comps = np.concatenate([rx_iq.real, rx_iq.imag])
    peak = float(np.max(np.abs(comps))) if comps.size else 0.0
    rms = float(np.sqrt(np.mean(comps**2))) if comps.size else 0.0
    peak_lsb, rms_lsb = peak * 127.0, rms * 127.0
    warnings = []
    if peak_lsb >= 120:
        warnings.append(f"RX near/at saturation (peak {peak_lsb:.0f}/127 LSB) -- reduce RX gain or add attenuation")
    # Deliberately keyed to rms, not peak. At the bottom of an SNR sweep the
    # signal is SUPPOSED to be small; what decides whether the capture is still
    # honest is whether the noise dithers the converter. Warning on a small peak
    # instead fires on every low-SNR point -- exactly the ones the sweep needs.
    if rms_lsb < _DITHER_FLOOR_LSB:
        warnings.append(
            f"RX rms {rms_lsb:.2f} LSB is at the converter floor -- quantization noise will "
            f"dominate; raise RX gain"
        )
    return {
        "peak_fullscale": peak,
        "peak_lsb": peak_lsb,
        "rms_fullscale": rms,
        "rms_lsb": rms_lsb,
        "clipped": peak_lsb >= 120,
        "dc_offset": complex(np.mean(rx_iq)) if rx_iq.size else 0j,
        "warnings": warnings,
    }
