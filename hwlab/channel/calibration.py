"""Gain-setting <-> measured-SNR lookup, produced by scripts/calibrate_snr.py.

A commanded gain is not an SNR. The link's mapping between the two depends on
the attenuator, the cable, and the exact receive gains, so it is measured once
and then looked up. Every hardware plot should be labelled with the MEASURED
SNR recorded here, never with the nominal target.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from hwlab.config import GainConfig


@dataclass
class CalibrationPoint:
    tx_vga_db: float
    tx_amp: bool
    rx_lna_db: float
    rx_vga_db: float
    measured_snr_db: float
    snr_std_db: float
    bursts: int
    loss_rate: float
    peak_lsb: float
    warnings: list
    #: Structured disqualifiers, kept separate from `warnings` so that usability
    #: is decided by measured conditions rather than by matching English text.
    #: Defaulted so calibration files written before these existed still load.
    clipped: bool = False
    snr_disagreement_db: float = 0.0

    def gains(self) -> GainConfig:
        return GainConfig(
            tx_vga_db=self.tx_vga_db,
            tx_amp=self.tx_amp,
            rx_lna_db=self.rx_lna_db,
            rx_vga_db=self.rx_vga_db,
        )


@dataclass
class Calibration:
    points: list
    attenuator_db: float = 0.0
    note: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            points=[CalibrationPoint(**p) for p in raw["points"]],
            attenuator_db=raw.get("attenuator_db", 0.0),
            note=raw.get("note", ""),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "attenuator_db": self.attenuator_db,
                    "note": self.note,
                    "points": [asdict(p) for p in self.points],
                },
                f,
                indent=2,
            )

    #: Above this the guard and pilot noise estimates are measuring different
    #: things, so the recorded SNR cannot be trusted as an axis label.
    MAX_SNR_DISAGREEMENT_DB = 3.0

    def usable(self) -> list:
        """Points that are actually trustworthy: no clipping, no burst loss, and
        the two independent noise estimates agreeing.

        Note what is NOT disqualifying: a small received level. At the bottom of
        an SNR sweep the signal is supposed to be small, and rejecting those
        points would throw away exactly the region the experiment is about.
        Whether such a point is honest is decided by the converter-floor check in
        `measure.level_report`, which keys on rms rather than peak.
        """
        return [
            p
            for p in self.points
            if not p.clipped
            and p.loss_rate == 0.0
            and p.snr_disagreement_db <= self.MAX_SNR_DISAGREEMENT_DB
        ]

    def nearest(self, target_snr_db: float, require_usable: bool = True) -> CalibrationPoint:
        candidates = self.usable() if require_usable else self.points
        if not candidates:
            raise ValueError("calibration has no usable points -- re-run scripts/calibrate_snr.py")
        return min(candidates, key=lambda p: abs(p.measured_snr_db - target_snr_db))

    def achievable_range_db(self) -> tuple:
        snrs = [p.measured_snr_db for p in self.usable()]
        return (min(snrs), max(snrs)) if snrs else (float("nan"), float("nan"))
