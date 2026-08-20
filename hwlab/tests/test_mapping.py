# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Pins the hardware link's SNR convention against airComp's simulation channel.

If this file fails, every hardware-vs-simulation plot is off by a constant (in
practice 3 dB) and the comparison is meaningless. It is the single most
important test in `hwlab`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from airComp.channel.analog import AnalogAWGNChannel
from airComp.jscc.modules import SemanticEncoder
from hwlab.dsp.mapping import (
    SIGNAL_POWER_PER_REAL,
    complex_awgn,
    noise_var_per_real,
    symbols_to_z,
    z_to_symbols,
)

K = 16


def _normalized_z(rng: np.random.Generator, k: int = K) -> np.ndarray:
    z = rng.normal(size=k)
    return z * np.sqrt(k) / np.linalg.norm(z)


def test_symbol_mapping_round_trip_is_exact():
    rng = np.random.default_rng(0)
    z = _normalized_z(rng)
    assert np.allclose(symbols_to_z(z_to_symbols(z)), z, atol=0)


def test_encoder_output_has_unit_power_per_real_component():
    """The convention rests on ||z|| = sqrt(k); assert the encoder really does that."""
    encoder = SemanticEncoder(input_dim=64, k=K).eval()
    with torch.no_grad():
        z = encoder(torch.randn(32, 64)).numpy()
    per_real_power = np.mean(z**2, axis=1)
    assert np.allclose(per_real_power, SIGNAL_POWER_PER_REAL, atol=1e-5)


def test_symbol_power_is_two_not_one():
    """Guards the exact spot where a stray 1/sqrt(2) would introduce a 3 dB error."""
    rng = np.random.default_rng(1)
    z = np.concatenate([_normalized_z(rng) for _ in range(64)])
    symbols = z_to_symbols(z)
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 10.0, 20.0])
def test_noise_var_matches_simulation_channel(snr_db):
    """hwlab's noise_var_per_real == the variance AnalogAWGNChannel actually adds."""
    torch.manual_seed(0)
    z = torch.randn(4096, K)
    z = z * (K**0.5) / z.norm(dim=-1, keepdim=True)
    y = AnalogAWGNChannel()(z, snr_db)
    measured = float(((y - z) ** 2).mean())
    assert measured == pytest.approx(noise_var_per_real(snr_db), rel=0.05)


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 10.0, 20.0])
def test_complex_awgn_matches_simulation_channel_after_unmapping(snr_db):
    """The full complex path: z -> symbols -> complex AWGN -> z must have the
    same per-real error variance as the simulation channel at the same SNR."""
    rng = np.random.default_rng(2)
    z = np.stack([_normalized_z(rng) for _ in range(4096)])
    symbols = z_to_symbols(z)
    received = complex_awgn(symbols, noise_var_per_real(snr_db), rng)
    err = symbols_to_z(received) - z
    assert float(np.mean(err**2)) == pytest.approx(noise_var_per_real(snr_db), rel=0.05)
