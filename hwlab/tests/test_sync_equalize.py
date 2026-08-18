"""Frame detection and equalization under an ideal (noiseless) channel.

Any failure here is an index or convention bug in the DSP, not a channel
effect -- which is exactly why these run without noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from hwlab.config import BurstConfig, GainConfig, LinkConfig, LoopbackConfig
from hwlab.dsp.burst import BurstCodec
from hwlab.dsp.equalize import estimate_gain
from hwlab.radio.loopback import LoopbackBackend

UNITY_GAINS = GainConfig(tx_vga_db=0.0, tx_amp=False, rx_lna_db=0.0, rx_vga_db=0.0)

#: Even a noiseless link is not exact: truncating the RRC leaves a small ISI
#: floor (about -65 dB at the default span=24). test_loopback_e2e.py measures
#: that floor explicitly; here we just tolerate it.
ISI_TOL = 5e-3


def ideal_backend(**overrides) -> LoopbackBackend:
    cfg = LoopbackConfig(
        path_loss_db=0.0,
        noise_floor_dbfs=-300.0,
        dc_offset=0.0,
        quantize=False,
        **overrides,
    )
    backend = LoopbackBackend(cfg)
    backend.configure(UNITY_GAINS)
    return backend


def normalized_z(rng: np.random.Generator, k: int) -> np.ndarray:
    z = rng.normal(size=k)
    return z * np.sqrt(k) / np.linalg.norm(z)


def test_noiseless_round_trip_recovers_z():
    codec = BurstCodec(LinkConfig(), BurstConfig())
    z = normalized_z(np.random.default_rng(0), codec.k)
    rx = ideal_backend().send_and_capture(codec.modulate(z), codec.capture_samples)

    decoded = codec.demodulate(rx)
    assert decoded is not None, "preamble correlation failed on a noiseless channel"
    assert np.allclose(decoded.z_hat, z, atol=ISI_TOL)


@pytest.mark.parametrize("timing_offset", [0, 1, 613, 7919])
def test_sync_is_invariant_to_capture_offset(timing_offset):
    """No hardware trigger is used, so an arbitrary capture offset must be
    absorbed by preamble correlation alone."""
    codec = BurstCodec(LinkConfig(), BurstConfig())
    z = normalized_z(np.random.default_rng(1), codec.k)
    backend = ideal_backend(timing_offset_samples=timing_offset)

    decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
    assert decoded is not None
    assert np.allclose(decoded.z_hat, z, atol=ISI_TOL)


@pytest.mark.parametrize("phase_deg", [0.0, 37.0, 175.0, -120.0])
def test_equalizer_removes_arbitrary_phase(phase_deg):
    codec = BurstCodec(LinkConfig(), BurstConfig())
    z = normalized_z(np.random.default_rng(2), codec.k)
    backend = ideal_backend(phase_offset_deg=phase_deg)

    decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
    assert decoded is not None
    assert np.allclose(decoded.z_hat, z, atol=ISI_TOL)


def test_equalizer_removes_arbitrary_amplitude():
    """A 30 dB change in link gain must not change the recovered z at all --
    if it does, the decoder is being fed the wrong power scale."""
    codec = BurstCodec(LinkConfig(), BurstConfig())
    z = normalized_z(np.random.default_rng(3), codec.k)
    tx = codec.modulate(z)

    recovered = []
    for tx_gain in (0.0, 30.0):
        backend = ideal_backend()
        backend.configure(GainConfig(tx_vga_db=tx_gain, rx_lna_db=0.0, rx_vga_db=0.0))
        decoded = codec.demodulate(backend.send_and_capture(tx, codec.capture_samples))
        assert decoded is not None
        recovered.append(decoded.z_hat)

    assert np.allclose(recovered[0], z, atol=ISI_TOL)
    assert np.allclose(recovered[1], z, atol=ISI_TOL)


def test_dc_offset_is_rejected():
    """Zero-IF DC spur must not reach the data symbols (IF offset + DC removal)."""
    codec = BurstCodec(LinkConfig(), BurstConfig())
    z = normalized_z(np.random.default_rng(4), codec.k)
    backend = LoopbackBackend(
        LoopbackConfig(path_loss_db=0.0, noise_floor_dbfs=-300.0, dc_offset=0.05, quantize=False)
    )
    backend.configure(UNITY_GAINS)

    decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
    assert decoded is not None
    assert np.allclose(decoded.z_hat, z, atol=ISI_TOL)


def test_ls_gain_estimate_is_exact_without_noise():
    pilots = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j] * 8)
    h = 0.37 * np.exp(1j * 1.1)
    assert estimate_gain(pilots * h, pilots) == pytest.approx(h)


def test_returns_none_when_no_burst_present():
    codec = BurstCodec(LinkConfig(), BurstConfig())
    noise = np.random.default_rng(5).normal(size=codec.capture_samples) * 0.1
    assert codec.demodulate(noise.astype(complex)) is None
