"""Phase 1: is the link alive and correctly set up?

Run this FIRST, before calibration and before any sweep. It answers, in order:
  1. Are the two HackRFs clock-locked (CLKOUT -> CLKIN)?
  2. Does the preamble correlate -- i.e. is a burst getting through at all?
  3. Is the ADC in its sweet spot -- neither clipping nor down in the dirt?
  4. Do the two independent noise estimates agree?
  5. Does z survive the round trip?

Usage:
    python -m hwlab.scripts.check_link --backend loopback
    python -m hwlab.scripts.check_link --backend hackrf --tx-gain 30

SAFETY, before connecting anything: the HackRF transmits up to about +15 dBm and
its receiver is damaged above roughly -5 dBm. Put at least 30 dB of attenuation
between TX and RX, and never transmit into an open port.
"""
from __future__ import annotations

import argparse

import numpy as np

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
    parser.add_argument("--tx-gain", type=float, default=None, help="override gains.tx_vga_db")
    # RX gain overrides exist for bring-up only: this is where the ADC operating point
    # gets found. Once found, it belongs in the config and stays FIXED for a whole
    # sweep -- moving it mid-sweep moves the operating point, not the SNR.
    parser.add_argument("--rx-lna", type=float, default=None, help="override gains.rx_lna_db (0..40, 8 dB steps)")
    parser.add_argument("--rx-vga", type=float, default=None, help="override gains.rx_vga_db (0..62, 2 dB steps)")
    parser.add_argument("--bursts", type=int, default=10)
    args = parser.parse_args()

    return report_setup_problems(_run, args)


def _run(args) -> int:
    cfg = build_config(args)
    gains = GainConfig(**vars(cfg.gains))
    if args.tx_gain is not None:
        gains.tx_vga_db = args.tx_gain
    if args.rx_lna is not None:
        gains.rx_lna_db = args.rx_lna
    if args.rx_vga is not None:
        gains.rx_vga_db = args.rx_vga

    # Devices first: a unit held by a stranded process makes every later step fail
    # in a way that looks like a wiring fault. `with` closes it on every exit path.
    with build_backend(args, cfg) as backend:
        return _check(args, cfg, gains, backend)


def _check(args, cfg, gains: GainConfig, backend) -> int:
    print("== clock ==")
    print(check_clock(args, cfg))

    codec = BurstCodec(cfg.link, cfg.burst)
    print("\n== waveform ==")
    print(f"  symbol rate      {cfg.link.symbol_rate/1e3:.1f} ksym/s at {cfg.link.fs/1e6:.2f} Msps (sps={cfg.link.sps})")
    print(f"  occupied BW      {cfg.link.symbol_rate*(1+cfg.link.rolloff)/1e3:.1f} kHz at +{cfg.link.if_offset_hz/1e3:.0f} kHz IF")
    print(f"  burst            {codec.layout.total_symbols} symbols "
          f"({codec.layout.n_data} data + {codec.layout.overhead_symbols} overhead), "
          f"{codec.layout.total_symbols/cfg.link.symbol_rate*1e3:.2f} ms")
    print(f"  capture          {codec.capture_samples} samples")
    print(f"  gains            tx_vga={gains.tx_vga_db} amp={gains.tx_amp} "
          f"rx_lna={gains.rx_lna_db} rx_vga={gains.rx_vga_db}")

    backend.configure(gains)

    rng = np.random.default_rng(0)
    rows, lost = [], 0
    for _ in range(args.bursts):
        z = rng.normal(size=codec.k)
        z *= np.sqrt(codec.k) / np.linalg.norm(z)
        decoded = codec.demodulate(backend.send_and_capture(codec.modulate(z), codec.capture_samples))
        if decoded is None:
            lost += 1
            continue
        rows.append((decoded, float(np.mean((decoded.z_hat - z) ** 2))))

    print("\n== link ==")
    print(f"  bursts detected  {len(rows)}/{args.bursts}")
    if not rows:
        print("\n  NO BURST DETECTED. Check, in this order:")
        print("   - RF cable and attenuator actually connected TX -> RX")
        print("   - both radios on the same center frequency and sample rate")
        print("   - TX gain high enough / attenuation not excessive")
        print("   - CLKOUT -> CLKIN present (an unlocked pair will not sync)")
        return 1

    decoded = [r[0] for r in rows]
    errs = np.array([r[1] for r in rows])
    peak_lsb = np.mean([d.levels["peak_lsb"] for d in decoded])
    guard_snr = np.mean([d.snr_db for d in decoded])
    pilot_snr = np.mean([d.pilot_snr_db for d in decoded])

    print(f"  preamble peak    {np.mean([d.peak_ratio for d in decoded]):.1f}x median "
          f"(>{cfg.link.min_peak_ratio} required)")
    # Peak alone is the wrong criterion: at the bottom of an SNR sweep the signal
    # is supposed to be small. What has to hold is that nothing clips and that the
    # noise still dithers the converter -- so both numbers are shown.
    print(f"  RX level         peak {peak_lsb:.0f}/127 LSB (must stay under 120), "
          f"rms {np.mean([d.levels['rms_lsb'] for d in decoded]):.2f} LSB (must stay over 1)")
    print(f"  |h|              {np.mean([abs(d.h) for d in decoded]):.5f}")
    print(f"  image rejection  {np.mean([d.image_rejection_db for d in decoded]):.1f} dB")
    print("\n== SNR ==")
    print(f"  guard estimate   {guard_snr:+.2f} dB   (primary)")
    print(f"  pilot residual   {pilot_snr:+.2f} dB   (independent cross-check)")
    print(f"  from z error     {-10*np.log10(errs.mean()):+.2f} dB   (ground truth, loopback only)")

    ok = True
    if abs(guard_snr - pilot_snr) > 3.0:
        print("\n  WARNING: the two noise estimates disagree by more than 3 dB.")
        print("  Suspect, in order: RX clipping, a spur inside the signal band,")
        print("  or a timing error in frame detection.")
        ok = False
    for d in decoded[:1]:
        for w in d.levels["warnings"]:
            print(f"\n  WARNING: {w}")
            ok = False
    if lost:
        print(f"\n  WARNING: {lost} burst(s) lost. Inside the sweep range this must be zero,")
        print("  or sync failures will be recorded as semantic failures.")
        ok = False

    print("\n" + ("Link looks good. Next: python -m hwlab.scripts.calibrate_snr" if ok else "Fix the warnings above before calibrating."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
