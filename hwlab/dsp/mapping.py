# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""THE single definition of the SNR convention for the hardware link.

Everything else in `hwlab` defers to this module. Getting this wrong shifts the
whole hardware degradation curve by 3 dB and silently invalidates every
hardware-vs-simulation comparison, so it lives in one place with one test
(`hwlab/tests/test_mapping.py`) that pins it against the simulation channel.

The convention, matching `airComp/channel/analog.py`:

    SNR is defined PER REAL COMPONENT.

`SemanticEncoder` emits z with ||z|| = sqrt(k) (see airComp/jscc/modules.py),
so the average power of each real component of z is exactly 1.0.
`AnalogAWGNChannel` adds i.i.d. N(0, sigma^2) to each real component with
sigma^2 = 1/snr_linear. We therefore:

  * map z to complex symbols by taking consecutive pairs AS-IS:
        s = z[0::2] + 1j*z[1::2]
    which gives E|s|^2 = 2 (NOT 1). We deliberately do *not* rescale to unit
    symbol power -- that rescaling is exactly where the 3 dB error creeps in.

  * define measured SNR as (signal power per real component) divided by
    (noise power per real component).

With this convention the complex-domain Es/N0 happens to agree numerically
(Es = 2, N0 = 2*sigma^2), so there is no factor to remember -- but only as
long as neither side is renormalized by 1/sqrt(2).
"""
from __future__ import annotations

import numpy as np

#: Average power of one real component of a power-normalized z (||z||^2 = k).
SIGNAL_POWER_PER_REAL = 1.0


def z_to_symbols(z: np.ndarray) -> np.ndarray:
    """(..., k) real -> (..., k/2) complex. k must be even."""
    z = np.asarray(z)
    if z.shape[-1] % 2 != 0:
        raise ValueError(f"k must be even to pair into complex symbols, got k={z.shape[-1]}")
    return z[..., 0::2] + 1j * z[..., 1::2]


def symbols_to_z(symbols: np.ndarray) -> np.ndarray:
    """(..., k/2) complex -> (..., k) real. Exact inverse of z_to_symbols."""
    symbols = np.asarray(symbols)
    out = np.empty(symbols.shape[:-1] + (2 * symbols.shape[-1],), dtype=float)
    out[..., 0::2] = symbols.real
    out[..., 1::2] = symbols.imag
    return out


def noise_var_per_real(snr_db: float, signal_power_per_real: float = SIGNAL_POWER_PER_REAL) -> float:
    """Noise variance per real component for a target SNR."""
    return signal_power_per_real / (10.0 ** (snr_db / 10.0))


def snr_db_from_powers(signal_power_per_real: float, noise_power_per_real: float) -> float:
    if noise_power_per_real <= 0:
        return float("inf")
    return 10.0 * np.log10(signal_power_per_real / noise_power_per_real)


def complex_awgn(x: np.ndarray, noise_var_per_real_: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN whose *per-real-component* variance is `noise_var_per_real_`.

    Note the total complex variance is 2x that value -- this is the convention
    boundary, and the reason this helper exists instead of inline randn calls.
    """
    sigma = float(np.sqrt(noise_var_per_real_))
    noise = rng.normal(0.0, sigma, x.shape) + 1j * rng.normal(0.0, sigma, x.shape)
    return x + noise
