"""The path probe has to be trustworthy in absolute terms, not just relative.

Its whole purpose is to say "the link is N dB short", and a biased estimator
would send the operator after the wrong piece of hardware. So the tone estimator
is checked against a known amplitude, and the end-to-end measurement against the
loopback model's closed-form gain.
"""
from __future__ import annotations

import numpy as np
import pytest

from hwlab.config import GainConfig, HwConfig, LoopbackConfig
from hwlab.dsp.burst import BurstCodec
from hwlab.radio.loopback import LoopbackBackend
from hwlab.scripts.check_path import (
    CAPTURE_SAMPLES,
    clock_warning,
    find_tone,
    make_tone,
    noise_verdict,
    reference_gain_db,
    rx_report,
)


@pytest.fixture
def cfg():
    return HwConfig()


# --- tone generation ---------------------------------------------------------


def test_tone_survives_being_repeated(cfg):
    """Both backends transmit by repeating the block, so the measured amplitude
    must be the amplitude after tiling -- not just of a single block."""
    n = BurstCodec(cfg.link, cfg.burst).burst_samples
    tone, f = make_tone(cfg, n)
    tiled = np.tile(tone, int(np.ceil(CAPTURE_SAMPLES / n)))[:CAPTURE_SAMPLES]

    amp, _, detected = find_tone(tiled, cfg.link.fs, f)

    assert detected
    assert amp == pytest.approx(cfg.link.dac_peak, rel=0.01)
    # Snapped, not moved: at most half a bin of the block (here exactly half).
    assert abs(f - cfg.link.if_offset_hz) <= cfg.link.fs / (2 * n) * (1 + 1e-9)


def test_a_block_that_does_not_close_loses_amplitude_when_repeated(cfg):
    """Why make_tone snaps the frequency. The naive tone at exactly if_offset_hz
    does not close on a whole cycle over a burst-length block, so every repeat
    adds a phase jump -- which reads as path loss that is not there."""
    n = BurstCodec(cfg.link, cfg.burst).burst_samples
    naive = cfg.link.dac_peak * np.exp(2j * np.pi * cfg.link.if_offset_hz * np.arange(n) / cfg.link.fs)
    tiled = np.tile(naive, int(np.ceil(CAPTURE_SAMPLES / n)))[:CAPTURE_SAMPLES]

    amp, _, _ = find_tone(tiled, cfg.link.fs, cfg.link.if_offset_hz)

    assert amp < 0.7 * cfg.link.dac_peak  # measured ~4 dB of phantom loss


# --- tone estimator ----------------------------------------------------------


@pytest.mark.parametrize("amplitude", [0.8, 0.01, 1e-4, 1e-6])
def test_find_tone_is_unbiased_over_120_db(amplitude, cfg):
    """The probe reports absolute path gain, so a scale error anywhere in the
    dynamic range sends the operator after the wrong piece of hardware."""
    rng = np.random.default_rng(0)
    n = 1 << 15
    f = 250_000.0
    x = amplitude * np.exp(2j * np.pi * f * np.arange(n) / cfg.link.fs)
    x = x + amplitude / 50.0 * (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n))

    amp, foff, detected = find_tone(x, cfg.link.fs, f)

    assert detected
    assert amp == pytest.approx(amplitude, rel=0.02)
    assert abs(foff) < 100.0


def test_find_tone_still_works_well_below_the_sample_level_noise(cfg):
    """Coherent integration over the whole capture is what lets this probe
    measure a link far too weak for preamble correlation to see."""
    rng = np.random.default_rng(0)
    n = CAPTURE_SAMPLES
    f = 250_000.0
    tone = 1e-3 * np.exp(2j * np.pi * f * np.arange(n) / cfg.link.fs)
    noise = rng.normal(0, 1e-2, n) + 1j * rng.normal(0, 1e-2, n)  # 20 dB above the tone

    amp, _, detected = find_tone(tone + noise, cfg.link.fs, f)

    assert detected
    assert amp == pytest.approx(1e-3, rel=0.15)


def test_find_tone_reports_not_detected_in_pure_noise(cfg):
    rng = np.random.default_rng(0)
    n = 1 << 15
    noise = rng.normal(0, 1e-3, n) + 1j * rng.normal(0, 1e-3, n)

    _, _, detected = find_tone(noise, cfg.link.fs, 250_000.0)

    assert not detected


def test_find_tone_follows_an_offset_carrier(cfg):
    """An unlocked pair -- the normal state when --both-directions reverses the
    roles -- offsets the tone by the crystal difference, several kHz at 915 MHz."""
    n = 1 << 15
    f = 250_000.0 + 7_000.0
    x = 0.5 * np.exp(2j * np.pi * f * np.arange(n) / cfg.link.fs)

    amp, foff, detected = find_tone(x, cfg.link.fs, 250_000.0)

    assert detected
    assert amp == pytest.approx(0.5, rel=0.05)
    assert foff == pytest.approx(7_000.0, abs=100.0)


# --- end-to-end against the model's closed form ------------------------------


@pytest.mark.parametrize("tx_gain, path_loss", [(30.0, 90.0), (40.0, 90.0), (30.0, 70.0)])
def test_measured_path_gain_matches_the_closed_form(tx_gain, path_loss, cfg):
    """tx_vga - path_loss + rx_lna + rx_vga, measured rather than asserted."""
    cfg.loopback = LoopbackConfig(path_loss_db=path_loss, noise_floor_dbfs=-90.0, quantize=False)

    measured = reference_gain_db(cfg, tx_gain)

    expected = tx_gain - path_loss + cfg.gains.rx_lna_db + cfg.gains.rx_vga_db
    assert measured == pytest.approx(expected, abs=0.2)


# --- receiver characterization ----------------------------------------------


def test_capture_only_carries_no_tone(cfg):
    """--noise-only must measure the receiver, not a residue of the transmitter."""
    backend = LoopbackBackend(LoopbackConfig(noise_floor_dbfs=-60.0, quantize=False))
    backend.configure(GainConfig())

    _, _, detected = find_tone(backend.capture_only(CAPTURE_SAMPLES), cfg.link.fs, cfg.link.if_offset_hz)

    assert not detected


def test_a_free_running_pair_is_called_out():
    """A burst is 8.71 ms long, so even a few hundred Hz rotates it through whole
    cycles. The measured free-running offset on this bench was -24 kHz."""
    assert "NOT sharing a reference" in clock_warning([-24_000.0, -24_010.0])
    assert clock_warning([120.0, -200.0]) == ""
    assert clock_warning([]) == ""


def _row(total_gain_db, rms, peak, peak_over_rms=None):
    return (total_gain_db, {"rms_lsb": rms, "peak_lsb": peak,
                            "peak_over_rms": peak_over_rms if peak_over_rms is not None else peak / rms})


def test_noise_that_ignores_receive_gain_is_called_digital():
    """The real measurement from the clone unit: 44 dB of receive gain removed,
    noise unmoved, peak pinned at full scale. Analog noise cannot do that, so it
    is being generated after the gain stages and no gain setting will help."""
    rows = [_row(44.0, 4.295, 128), _row(32.0, 4.232, 128), _row(16.0, 4.223, 128), _row(0.0, 3.855, 128)]

    verdict, gain_span, rms_span = noise_verdict(rows)

    assert verdict == "digital"
    assert gain_span == pytest.approx(44.0)
    assert abs(rms_span) < 1.0


def test_noise_that_tracks_receive_gain_is_not_called_digital():
    """A healthy receiver: 44 dB less gain gives 44 dB less noise."""
    rows = [_row(44.0, 40.0, 180), _row(32.0, 10.0, 45), _row(16.0, 1.6, 7), _row(0.0, 0.25, 1.2)]

    assert noise_verdict(rows)[0] == "clean"


def test_impulsive_but_analog_noise_is_reported_separately():
    """External interference does scale with gain -- it comes in the antenna
    port -- so it must not be blamed on the radio's digital side."""
    rows = [_row(44.0, 4.0, 128, peak_over_rms=32.0), _row(32.0, 1.0, 32), _row(16.0, 0.16, 5), _row(0.0, 0.025, 1)]

    assert noise_verdict(rows)[0] == "impulsive"


def test_rx_report_separates_impulsive_from_thermal():
    """peak/rms is the discriminator the noise-only mode warns on."""
    rng = np.random.default_rng(0)
    n = 200_000
    thermal = (rng.normal(0, 0.03, n) + 1j * rng.normal(0, 0.03, n))
    impulsive = thermal.copy()
    impulsive[rng.integers(0, n, 20)] = 1.0 + 1.0j

    assert rx_report(thermal)["peak_over_rms"] < 10.0
    assert rx_report(impulsive)["peak_over_rms"] > 10.0
    assert rx_report(impulsive)["p999_lsb"] > rx_report(thermal)["p999_lsb"]
