"""Phase 2: measure the gain-setting -> SNR mapping and write a calibration table.

A commanded TX gain is not an SNR. This sweeps TX gain with the RX gains held
fixed, measures the SNR actually delivered at each setting, and records it.
Everything downstream looks up settings here and labels its plots with the
measured value.

Points that clip the ADC or lose bursts are recorded but flagged, and
`Calibration.usable()` excludes them -- a clipped point reports a rising SNR
while the actual error gets worse, which is the most dangerous way for this
experiment to go quietly wrong.

Usage:
    python -m hwlab.scripts.calibrate_snr --backend loopback
    python -m hwlab.scripts.calibrate_snr --backend hackrf --attenuator-db 40 --bursts 20
"""
from __future__ import annotations

import argparse

import numpy as np

from hwlab.channel.calibration import Calibration, CalibrationPoint
from hwlab.config import GainConfig
from hwlab.dsp.burst import BurstCodec
from hwlab.scripts._common import (
    add_common_args,
    build_backend,
    build_config,
    check_clock,
    report_setup_problems,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--tx-gains", type=float, nargs="+", default=None,
                        help="TX VGA settings in dB (default: 0..47 in 2 dB steps)")
    parser.add_argument("--bursts", type=int, default=20, help="bursts averaged per gain setting")
    parser.add_argument("--attenuator-db", type=float, default=0.0,
                        help="external attenuation in the coax path; recorded, not applied")
    parser.add_argument("--note", default="", help="free text stored in the calibration file")
    parser.add_argument("--out", default=None, help="default: config's calibration_path")
    args = parser.parse_args()
    return report_setup_problems(_run, args)


def _run(args) -> int:
    cfg = build_config(args)
    out_path = args.out or cfg.calibration_path or "hwlab/results/calibration.json"
    tx_gains = args.tx_gains if args.tx_gains is not None else list(np.arange(0, 48, 2.0))

    # Devices before the clock check: hackrf_clock fails just as opaquely as
    # hackrf_transfer when a unit is still held by a stranded process.
    backend = build_backend(args, cfg)

    points = []
    try:
        print(check_clock(args, cfg))
        codec = BurstCodec(cfg.link, cfg.burst)
        rng = np.random.default_rng(0)

        print(f"\nRX gains held fixed at lna={cfg.gains.rx_lna_db} vga={cfg.gains.rx_vga_db} "
              f"(external attenuation {args.attenuator_db} dB)\n")
        print(f"{'tx_vga':>7} {'SNR dB':>8} {'sd':>6} {'peak':>6} {'lost':>6}  notes")

        for tx_gain in tx_gains:
            gains = GainConfig(
                tx_vga_db=float(tx_gain),
                tx_amp=cfg.gains.tx_amp,
                rx_lna_db=cfg.gains.rx_lna_db,
                rx_vga_db=cfg.gains.rx_vga_db,
            )
            backend.configure(gains)
            snrs, peaks, notes, lost = [], [], set(), 0
            clipped, disagreements = False, []
            for _ in range(args.bursts):
                z = rng.normal(size=codec.k)
                z *= np.sqrt(codec.k) / np.linalg.norm(z)
                decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
                if decoded is None:
                    lost += 1
                    continue
                snrs.append(decoded.snr_db)
                peaks.append(decoded.levels["peak_lsb"])
                notes.update(decoded.levels["warnings"])
                clipped = clipped or decoded.levels["clipped"]
                disagreements.append(decoded.snr_disagreement_db)
                if decoded.snr_disagreement_db > Calibration.MAX_SNR_DISAGREEMENT_DB:
                    notes.add(f"guard/pilot SNR disagree by {decoded.snr_disagreement_db:.1f} dB")

            loss_rate = lost / args.bursts
            if lost:
                notes.add(f"{lost}/{args.bursts} bursts lost")
            point = CalibrationPoint(
                tx_vga_db=float(tx_gain),
                tx_amp=bool(cfg.gains.tx_amp),
                rx_lna_db=float(cfg.gains.rx_lna_db),
                rx_vga_db=float(cfg.gains.rx_vga_db),
                measured_snr_db=float(np.mean(snrs)) if snrs else float("nan"),
                snr_std_db=float(np.std(snrs)) if snrs else float("nan"),
                bursts=len(snrs),
                loss_rate=float(loss_rate),
                peak_lsb=float(np.mean(peaks)) if peaks else float("nan"),
                warnings=sorted(notes),
                clipped=bool(clipped),
                # The worst burst decides: one capture where the two estimators
                # disagree means the recorded SNR is not a safe axis label.
                snr_disagreement_db=float(max(disagreements)) if disagreements else float("inf"),
            )
            points.append(point)
            print(f"{tx_gain:7.0f} {point.measured_snr_db:8.2f} {point.snr_std_db:6.2f} "
                  f"{point.peak_lsb:6.0f} {loss_rate:6.0%}  {'; '.join(point.warnings)}")
    finally:
        backend.close()

    calibration = Calibration(points=points, attenuator_db=args.attenuator_db, note=args.note)
    calibration.save(out_path)

    lo, hi = calibration.achievable_range_db()
    print(f"\nwrote {out_path}")
    print(f"usable points: {len(calibration.usable())}/{len(points)}")
    print(f"achievable SNR range: {lo:+.1f} .. {hi:+.1f} dB")
    if not (lo <= -10.0 and hi >= 20.0):
        print("\n  NOTE: the -10..+20 dB sweep range is not fully covered.")
        print("   - too little range at the bottom: add attenuation")
        print("   - too little at the top: remove attenuation (watch the RX peak level)")
        print("   - clipping at the top: lower rx_vga_db, then re-run this script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
