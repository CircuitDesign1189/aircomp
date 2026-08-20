# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Config loading, calibration persistence, and hackrf_transfer command building.

The HackRF command construction is tested without a radio: getting a gain flag
wrong is silent -- the radio just applies something else -- so it is pinned here
rather than discovered during a lab session.
"""
from __future__ import annotations

import numpy as np
import pytest

from hwlab.channel.calibration import Calibration, CalibrationPoint
from hwlab.config import DeviceConfig, GainConfig, HwConfig, LinkConfig, load_hw_config
from hwlab.radio.hackrf_cli import HackRFCLIBackend, HackRFError, _check_gain_steps, probe_devices
from hwlab.scripts._common import clock_pair


@pytest.fixture
def hackrf(monkeypatch, tmp_path):
    monkeypatch.setattr("hwlab.radio.hackrf_cli.shutil.which", lambda _: "hackrf_transfer")
    device = DeviceConfig(tx_serial="AAA", rx_serial="BBB", workdir=str(tmp_path))
    return HackRFCLIBackend(LinkConfig(), device)


# --- config -----------------------------------------------------------------


def test_shipped_config_loads_with_numeric_types():
    """PyYAML is YAML 1.1: `2.0e6` parses as a STRING. The loader must coerce,
    or fs silently becomes a str and fails several layers away."""
    cfg = load_hw_config("hwlab/configs/sdr_link.yaml")
    assert isinstance(cfg.link.fs, float)
    assert cfg.link.symbol_rate == pytest.approx(100_000.0)
    assert cfg.burst.n_data * 2 == 16  # matches JSCCConfig.k
    assert isinstance(cfg.gains.tx_amp, bool)


def test_unquoted_exponent_is_coerced(tmp_path):
    path = tmp_path / "link.yaml"
    path.write_text("link:\n  fs: 2.0e6\n  if_offset_hz: 250.0e3\n", encoding="utf-8")
    cfg = load_hw_config(path)
    assert cfg.link.fs == pytest.approx(2e6)
    assert cfg.link.if_offset_hz == pytest.approx(250e3)


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "link.yaml"
    path.write_text("link:\n  sample_rate: 2000000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key 'sample_rate'"):
        load_hw_config(path)


# --- calibration ------------------------------------------------------------


def test_calibration_round_trips(tmp_path):
    original = Calibration(
        points=[CalibrationPoint(30.0, False, 24.0, 20.0, 9.5, 0.2, 20, 0.0, 52.0, [])],
        attenuator_db=40.0,
        note="bench",
    )
    original.save(tmp_path / "cal.json")
    loaded = Calibration.load(tmp_path / "cal.json")

    assert loaded.attenuator_db == 40.0
    assert loaded.points[0].gains() == GainConfig(tx_vga_db=30.0, tx_amp=False, rx_lna_db=24.0, rx_vga_db=20.0)


def test_calibration_with_no_usable_points_raises():
    calibration = Calibration(points=[
        CalibrationPoint(46.0, False, 24.0, 20.0, 25.0, 0.3, 20, 0.0, 127.0, ["clip"], clipped=True)
    ])
    with pytest.raises(ValueError, match="no usable points"):
        calibration.nearest(20.0)


# --- HackRF gain validation -------------------------------------------------


@pytest.mark.parametrize(
    "gains, message",
    [
        (GainConfig(tx_vga_db=50.0), "outside 0..47"),
        (GainConfig(tx_vga_db=20.5), "not a whole dB"),
        (GainConfig(rx_lna_db=25.0), "8 dB steps"),
        (GainConfig(rx_vga_db=21.0), "2 dB steps"),
    ],
)
def test_out_of_step_gains_are_rejected(gains, message):
    """The radio silently rounds to its hardware steps; a calibration table that
    records a gain the radio never applied is worse than an error."""
    with pytest.raises(HackRFError, match=message):
        _check_gain_steps(gains)


def test_valid_gains_accepted():
    _check_gain_steps(GainConfig(tx_vga_db=30.0, rx_lna_db=24.0, rx_vga_db=20.0))


# --- command construction ---------------------------------------------------


def test_tx_command_is_finite_and_selects_the_tx_device(hackrf, tmp_path):
    hackrf.configure(GainConfig(tx_vga_db=30.0, tx_amp=False))
    cmd = hackrf._tx_cmd(tmp_path / "tx.bin", 3_053_700)

    assert cmd[cmd.index("-d") + 1] == "AAA"
    assert cmd[cmd.index("-x") + 1] == "30"
    assert cmd[cmd.index("-a") + 1] == "0"
    assert cmd[cmd.index("-f") + 1] == "915000000"
    assert cmd[cmd.index("-s") + 1] == "2000000"
    assert cmd[cmd.index("-n") + 1] == "3053700"
    assert "-R" not in cmd, (
        "infinite repeat has to be killed, and on Windows that strands the unit until it "
        "is replugged -- the transmit file is sized to end on its own instead"
    )


def test_tx_file_holds_whole_bursts_and_outlasts_the_capture(hackrf, tmp_path):
    """Repeat lives in the FILE now, so it must still cover the whole capture --
    that is what removes the need for a hardware trigger."""
    burst = np.zeros(17_420, dtype=complex)
    total = 253_700
    path = tmp_path / "tx.bin"

    written = hackrf._write_tx_file(burst, total, path)

    assert written % len(burst) == 0, "a truncated burst would never correlate"
    assert written >= hackrf.link.fs * (hackrf.tx_settle_s + hackrf.tx_margin_s) + total
    assert path.stat().st_size == 2 * written  # interleaved int8 I/Q


def test_tx_margin_is_reachable_from_the_config(monkeypatch, tmp_path):
    """The 'transmitter ran out of samples' error tells the operator to raise
    tx_margin_s, so it has to be settable without editing code."""
    monkeypatch.setattr("hwlab.radio.hackrf_cli.shutil.which", lambda _: "hackrf_transfer")
    device = DeviceConfig(workdir=str(tmp_path), tx_margin_s=3.0)
    backend = HackRFCLIBackend(LinkConfig(), device)

    written = backend._write_tx_file(np.zeros(1000, dtype=complex), 0, tmp_path / "tx.bin")
    assert written >= LinkConfig().fs * (0.4 + 3.0)


def test_empty_tx_is_rejected(hackrf, tmp_path):
    with pytest.raises(HackRFError, match="empty"):
        hackrf._write_tx_file(np.zeros(0, dtype=complex), 1000, tmp_path / "tx.bin")


def test_rx_command_sets_fixed_gains_and_sample_count(hackrf, tmp_path):
    hackrf.configure(GainConfig(rx_lna_db=24.0, rx_vga_db=20.0))
    cmd = hackrf._rx_cmd(tmp_path / "rx.bin", 53_700)

    assert cmd[cmd.index("-d") + 1] == "BBB"
    assert cmd[cmd.index("-l") + 1] == "24"
    assert cmd[cmd.index("-g") + 1] == "20"
    assert cmd[cmd.index("-n") + 1] == "53700"
    assert cmd[cmd.index("-a") + 1] == "0", "RX RF amp must stay off; SNR is set by TX gain and the attenuator"


def test_missing_hackrf_transfer_says_what_to_do(monkeypatch):
    monkeypatch.setattr("hwlab.radio.hackrf_cli.shutil.which", lambda _: None)
    with pytest.raises(HackRFError, match="LoopbackBackend"):
        HackRFCLIBackend(LinkConfig(), DeviceConfig())


# --- clock master/slave ------------------------------------------------------


def test_clock_master_defaults_to_the_transmitter():
    """Back-compatible with the README wiring: TX CLKOUT -> RX CLKIN."""
    cfg = HwConfig(device=DeviceConfig(tx_serial="AAA", rx_serial="BBB"))
    assert clock_pair(cfg) == ("AAA", "BBB")


def test_clock_direction_is_independent_of_the_transmit_direction():
    """A marginal CLKIN corrupts the slave's samples, and the fix is to reverse
    the CLOCK cable while leaving the TX/RX roles alone -- so the two must be
    expressible separately."""
    cfg = HwConfig(device=DeviceConfig(tx_serial="AAA", rx_serial="BBB", clock_master_serial="BBB"))
    assert clock_pair(cfg) == ("BBB", "AAA")


# --- device preflight -------------------------------------------------------
#
# A unit held by a stranded process reports itself as present-but-unopenable.
# Detecting that up front is the difference between a one-line fix and chasing
# a phantom wiring fault, so the parsing is pinned here.

_BOTH_OK = """\
hackrf_info version: 2024.02.1
Found HackRF
Index: 0
Serial number: 0000000000000000aaaa
Board ID Number: 4 (HackRF One)

Found HackRF
Index: 1
Serial number: 0000000000000000bbbb
Board ID Number: 4 (HackRF One)
"""

_ONE_CLAIMED = """\
hackrf_open() failed: Access denied (insufficient permissions) (-1000)
hackrf_info version: 2024.02.1
Found HackRF
Index: 0

Found HackRF
Index: 1
Serial number: 0000000000000000bbbb
Board ID Number: 4 (HackRF One)
"""


def _fake_info(monkeypatch, output: str):
    monkeypatch.setattr("hwlab.radio.hackrf_cli.shutil.which", lambda _: "hackrf_info")
    monkeypatch.setattr(
        "hwlab.radio.hackrf_cli.subprocess.run",
        lambda *a, **k: subprocess_result(output),
    )


class subprocess_result:  # noqa: N801 - stand-in for CompletedProcess
    def __init__(self, stdout: str):
        self.stdout, self.stderr, self.returncode = stdout, "", 0


def test_probe_reports_a_claimed_unit_with_no_serial(monkeypatch):
    _fake_info(monkeypatch, _ONE_CLAIMED)
    assert probe_devices() == [(0, ""), (1, "0000000000000000bbbb")]


def test_preflight_passes_when_both_serials_are_readable(monkeypatch, tmp_path):
    _fake_info(monkeypatch, _BOTH_OK)
    device = DeviceConfig(
        tx_serial="0000000000000000aaaa", rx_serial="0000000000000000bbbb", workdir=str(tmp_path)
    )
    assert "2 HackRF(s)" in HackRFCLIBackend(LinkConfig(), device).preflight()


def test_preflight_names_the_stranded_process_case(monkeypatch, tmp_path):
    _fake_info(monkeypatch, _ONE_CLAIMED)
    device = DeviceConfig(
        tx_serial="0000000000000000aaaa", rx_serial="0000000000000000bbbb", workdir=str(tmp_path)
    )
    with pytest.raises(HackRFError, match="replugging"):
        HackRFCLIBackend(LinkConfig(), device).preflight()


def test_preflight_rejects_two_roles_on_one_radio(monkeypatch, tmp_path):
    _fake_info(monkeypatch, _BOTH_OK)
    device = DeviceConfig(tx_serial="", rx_serial="", workdir=str(tmp_path))
    with pytest.raises(HackRFError, match="same unit|both transmit and receive"):
        HackRFCLIBackend(LinkConfig(), device).preflight()


def test_preflight_says_which_serial_is_absent(monkeypatch, tmp_path):
    _fake_info(monkeypatch, _BOTH_OK)
    device = DeviceConfig(tx_serial="deadbeef", rx_serial="0000000000000000bbbb", workdir=str(tmp_path))
    with pytest.raises(HackRFError, match="tx_serial=deadbeef is not attached"):
        HackRFCLIBackend(LinkConfig(), device).preflight()
