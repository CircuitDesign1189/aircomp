# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""End-to-end link tests over the loopback backend. No hardware required.

These are the tests that decide whether the hardware results will be
trustworthy. In particular `test_link_reproduces_simulation_channel` is the
software half of the plan's headline verification: if the measured SNR label is
truthful, the hardware degradation curve can be overlaid on the pure-simulation
curve and any systematic offset is a bug rather than a physical effect.
"""
from __future__ import annotations

import numpy as np
import pytest

from hwlab.config import BurstConfig, GainConfig, LinkConfig, LoopbackConfig
from hwlab.dsp.burst import BurstCodec
from hwlab.dsp.mapping import noise_var_per_real
from hwlab.radio.loopback import LoopbackBackend

#: tx_vga_db values that map to roughly -10, 0, +10, +20 dB measured SNR with
#: the default LoopbackConfig. Kept small so the suite stays quick.
SWEEP_POINTS = [10.0, 20.0, 30.0, 40.0]
BURSTS_PER_POINT = 40


def _codec() -> BurstCodec:
    return BurstCodec(LinkConfig(), BurstConfig())


def _random_z(rng: np.random.Generator, k: int) -> np.ndarray:
    z = rng.normal(size=k)
    return z * np.sqrt(k) / np.linalg.norm(z)


def _run(codec: BurstCodec, backend: LoopbackBackend, n: int, seed: int = 0):
    """Returns (reported_snr_db, per-burst error power per real, n_lost)."""
    rng = np.random.default_rng(seed)
    snrs, errs, lost = [], [], 0
    for _ in range(n):
        z = _random_z(rng, codec.k)
        decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
        if decoded is None:
            lost += 1
            continue
        snrs.append(decoded.snr_db)
        errs.append(float(np.mean((decoded.z_hat - z) ** 2)))
    return np.array(snrs), np.array(errs), lost


def _backend(tx_vga_db: float, seed: int = 0, **cfg) -> LoopbackBackend:
    backend = LoopbackBackend(LoopbackConfig(seed=seed, **cfg))
    backend.configure(GainConfig(tx_vga_db=tx_vga_db))
    return backend


# --- the DSP's own error floor ----------------------------------------------


def test_isi_floor_is_far_below_the_sweep_range():
    """A truncated RRC is not exactly ISI-free. The residual must stay far below
    the top of the sweep (+20 dB), or it -- not the channel -- sets the ceiling."""
    codec = _codec()
    backend = _backend(0.0, path_loss_db=0.0, noise_floor_dbfs=-300.0, dc_offset=0.0, quantize=False)
    _, errs, lost = _run(codec, backend, 8)
    assert lost == 0
    isi_floor_db = 10 * np.log10(errs.mean())
    assert isi_floor_db < -55.0, f"ISI floor {isi_floor_db:.1f} dB -- increase LinkConfig.span_symbols"


# --- the SNR label ----------------------------------------------------------


@pytest.mark.parametrize("tx_vga_db", SWEEP_POINTS)
def test_reported_snr_matches_the_error_it_predicts(tx_vga_db):
    """The x-axis of every hardware plot is this number. It must be truthful.

    Compared per burst and then averaged: averaging errors and averaging dB are
    not the same operation, and mixing them up looks like a 3 dB bug.
    """
    codec = _codec()
    snrs, errs, _ = _run(codec, _backend(tx_vga_db, seed=int(tx_vga_db)), BURSTS_PER_POINT)
    predicted = np.array([noise_var_per_real(s) for s in snrs])
    bias_db = float(np.mean(10 * np.log10(predicted / errs)))
    assert abs(bias_db) < 1.5, f"reported SNR is biased by {bias_db:+.2f} dB at tx_vga={tx_vga_db}"


def test_link_reproduces_simulation_channel():
    """The whole point: at a given measured SNR the link must add exactly the
    noise `AnalogAWGNChannel` would add at that SNR. If this drifts, hardware
    and simulation curves cannot be compared."""
    codec = _codec()
    snrs, errs, _ = _run(codec, _backend(30.0, seed=3), BURSTS_PER_POINT)
    assert float(np.mean(errs)) == pytest.approx(noise_var_per_real(float(np.mean(snrs))), rel=0.35)


def test_snr_tracks_tx_gain_one_for_one():
    codec = _codec()
    measured = [float(_run(codec, _backend(g, seed=int(g)), 12)[0].mean()) for g in SWEEP_POINTS]
    slope = np.polyfit(SWEEP_POINTS, measured, 1)[0]
    assert slope == pytest.approx(1.0, abs=0.1)
    assert all(b > a for a, b in zip(measured, measured[1:]))


# --- operational lessons the hardware will enforce --------------------------


def test_rx_gain_changes_adc_level_but_not_snr():
    """RX gain amplifies signal and noise together. It must be held fixed across
    a sweep -- not because it changes SNR, but because changing it changes the
    ADC operating point and can push the receiver into clipping."""
    codec = _codec()
    results = {}
    for rx_vga in (10.0, 30.0):
        backend = LoopbackBackend(LoopbackConfig(seed=1))
        backend.configure(GainConfig(tx_vga_db=30.0, rx_lna_db=24.0, rx_vga_db=rx_vga))
        z = _random_z(np.random.default_rng(0), codec.k)
        decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
        assert decoded is not None
        results[rx_vga] = decoded

    assert results[10.0].snr_db == pytest.approx(results[30.0].snr_db, abs=1.0)
    assert results[30.0].levels["peak_lsb"] > 5 * results[10.0].levels["peak_lsb"]


def test_clipping_is_reported():
    codec = _codec()
    backend = LoopbackBackend(LoopbackConfig(seed=2))
    backend.configure(GainConfig(tx_vga_db=47.0, rx_lna_db=40.0, rx_vga_db=40.0))
    z = _random_z(np.random.default_rng(0), codec.k)
    decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
    assert decoded is not None
    assert any("saturation" in w for w in decoded.levels["warnings"])


def test_eight_bit_quantization_is_not_the_limit_in_range():
    """The 8-bit converters are the most-questioned part of using a HackRF here.
    At the top of the sweep, turning quantization off must barely move the SNR."""
    codec = _codec()
    quantized, _, _ = _run(codec, _backend(40.0, seed=4, quantize=True), 12)
    ideal, _, _ = _run(codec, _backend(40.0, seed=4, quantize=False), 12)
    assert float(ideal.mean() - quantized.mean()) < 1.0


# --- robustness -------------------------------------------------------------


def test_no_burst_loss_across_the_sweep_range():
    """Frame detection must not fail inside the sweep. A sync failure would be
    recorded as a semantic failure and fake a cliff at low SNR."""
    codec = _codec()
    for tx_vga_db in SWEEP_POINTS:
        _, _, lost = _run(codec, _backend(tx_vga_db, seed=int(tx_vga_db) + 100), 25)
        assert lost == 0, f"lost {lost}/25 bursts at tx_vga={tx_vga_db}"


def test_pilot_cross_check_agrees_with_guard_estimate():
    """Two independent noise estimates -- guard region (pre-equalization, many
    samples) and pilot residual (post-equalization, measured where the data
    sits). Disagreement means clipping, an in-band spur, or a timing error."""
    codec = _codec()
    rng = np.random.default_rng(0)
    backend = _backend(30.0, seed=5)
    for _ in range(12):
        decoded = codec.demodulate(
            backend.send_and_capture(codec.modulate(_random_z(rng, codec.k)), codec.capture_samples)
        )
        assert decoded is not None
        assert decoded.snr_disagreement_db < 4.0
