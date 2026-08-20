# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Config dataclasses for the HackRF link. Mirrors the style of airComp/config.py."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml


@dataclass
class LinkConfig:
    """Waveform and capture parameters.

    Defaults: 2 Msps / 20 samples-per-symbol = 100 ksym/s, RRC rolloff 0.35, so
    the occupied bandwidth is 135 kHz sitting at a +250 kHz digital IF -- i.e.
    182..318 kHz, well clear of the zero-IF DC spur at baseband DC and well
    inside the 2 MHz capture band. Single carrier, not OFDM: with 8 complex
    symbols per message OFDM buys nothing, and its PAPR would eat headroom the
    8-bit converters cannot spare.
    """

    fs: float = 2.0e6
    sps: int = 20
    rolloff: float = 0.35
    #: RRC truncation length. A raised cosine is only ISI-free if untruncated.
    #: Measured end-to-end ISI floor (hwlab/tests/test_loopback_e2e.py):
    #: span=10 -> -43 dB, span=16 -> -54 dB, span=24 -> -65 dB. The sweep tops
    #: out at +20 dB SNR, so span=24 puts the ISI floor 85 dB down -- far below
    #: the 8-bit converters -- for 481 taps. Cheap insurance.
    span_symbols: int = 24
    if_offset_hz: float = 250.0e3
    center_freq_hz: float = 915.0e6
    dac_peak: float = 0.8  # fraction of full scale for the transmitted envelope
    capture_bursts: float = 3.0  # capture window, in burst lengths
    #: Preamble detection threshold, as correlation peak / median. Set from the
    #: measured separation between the two distributions over a ~50k-lag search:
    #:     pure noise, no signal : max 4.4
    #:     signal at -10 dB SNR  : min 6.7
    #: 4.0 sits inside the noise distribution and false-alarms on empty air --
    #: decoding noise as if it were a burst is worse than declaring a loss.
    min_peak_ratio: float = 5.5
    max_retries: int = 3

    @property
    def symbol_rate(self) -> float:
        return self.fs / self.sps


@dataclass
class BurstConfig:
    """Burst geometry. See hwlab/dsp/framing.py for what each section is for.

    preamble_len is set so that FRAME DETECTION IS NEVER THE LIMITING FACTOR
    inside the sweep range. Measured burst-loss rate on the loopback backend:

        preamble   -15 dB    -10 dB
        127        92%       12%
        511         2%        0%
        1023        0%        0%

    A sync failure at low SNR would show up in the results as a semantic
    failure, manufacturing exactly the cliff this project claims the semantic
    pipeline does not have. 511 gives a clean -10..+20 dB sweep; raise it to
    1023 (odd, coprime with root 25) before extending below -15 dB.
    """

    guard_symbols: int = 160
    preamble_len: int = 511
    zc_root: int = 25
    n_pilots: int = 32
    n_data: int = 8


@dataclass
class GainConfig:
    """HackRF gains. All manual -- the HackRF has no AGC, which is an advantage
    here: an AGC would silently rescale the received signal and break the power
    normalization SemanticDecoder depends on.

    RX gains must stay FIXED across an entire sweep. Move SNR with tx_vga_db and
    the external attenuator only.
    """

    tx_vga_db: float = 20.0  # 0..47, 1 dB steps
    tx_amp: bool = False  # +14 dB RF amp
    rx_lna_db: float = 24.0  # 0..40, 8 dB steps
    rx_vga_db: float = 20.0  # 0..62, 2 dB steps


@dataclass
class LoopbackConfig:
    """Impairment model for the no-hardware backend.

    Tuned so that, with the default GainConfig receive gains, tx_vga_db 10..40
    spans measured SNR -10..+20 dB -- the sweep range of interest -- with the
    ADC still in headroom. Above tx_vga_db ~44 the receiver clips, which the
    loopback reproduces faithfully: the guard-based SNR estimate keeps climbing
    while the actual error gets worse. That is the real link's most common
    failure mode, so it is modeled rather than idealized away.
    """

    path_loss_db: float = 90.0
    noise_floor_dbfs: float = -66.0  # per-real noise POWER at the RX input, in dBFS
    phase_offset_deg: float = 37.0
    timing_offset_samples: int = 613
    dc_offset: float = 0.01
    iq_gain_imbalance: float = 0.0
    iq_phase_deg: float = 0.0
    quantize: bool = True
    seed: int = 0


@dataclass
class DeviceConfig:
    tx_serial: str = ""  # empty -> let hackrf_transfer pick the only device
    rx_serial: str = ""
    #: Which unit drives CLKOUT. This is a property of the CABLE, not of who
    #: transmits: either unit can be the reference for either role. It is
    #: separate because a marginal CLKIN can corrupt the slave's samples, and
    #: then the fix is to reverse the clock cable while leaving TX/RX alone.
    #: Empty -> tx_serial, which is the CLKOUT -> CLKIN direction in the README.
    clock_master_serial: str = ""
    hackrf_transfer: str = "hackrf_transfer"
    hackrf_clock: str = "hackrf_clock"
    hackrf_info: str = "hackrf_info"
    workdir: str = ""  # empty -> a temp dir per run
    # Transmit timing. The transmit file is sized to cover settle + capture + margin so
    # that hackrf_transfer ends by itself; raise tx_margin_s if the receiver's start-up
    # is slow enough that the transmitter runs out first.
    tx_settle_s: float = 0.4
    tx_margin_s: float = 1.0


@dataclass
class HwConfig:
    link: LinkConfig = field(default_factory=LinkConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)
    gains: GainConfig = field(default_factory=GainConfig)
    loopback: LoopbackConfig = field(default_factory=LoopbackConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    calibration_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = ("link", "burst", "gains", "loopback", "device")


def _coerce(current, value):
    """Coerce a YAML value to the type of the dataclass default.

    PyYAML follows YAML 1.1, where `2.0e6` is a STRING -- the float resolver
    demands a signed exponent (`2.0e+6`). Silently accepting that would turn
    `fs` into a str and blow up several layers away, so types are pinned here.
    """
    if isinstance(current, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "yes", "1")
    for typ in (int, float):
        if isinstance(current, typ) and not isinstance(value, typ):
            return typ(value)
    if isinstance(current, str):
        return str(value)
    return value


def load_hw_config(path: str | Path | None) -> HwConfig:
    cfg = HwConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for name in _SECTIONS:
        section = raw.get(name)
        if not section:
            continue
        current = getattr(cfg, name)
        updates = {}
        for key, value in section.items():
            if not hasattr(current, key):
                raise ValueError(f"unknown key '{key}' in section '{name}' of {path}")
            updates[key] = _coerce(getattr(current, key), value)
        setattr(cfg, name, replace(current, **updates))
    if "calibration_path" in raw:
        cfg.calibration_path = str(raw["calibration_path"])
    return cfg
