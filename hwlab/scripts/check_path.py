# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.5: is the RF path itself healthy, and is the receiver quiet?

check_link answers "does a burst decode?", which is a yes/no that goes silent
exactly when you most need a number. This answers the two questions underneath
it, and it keeps working when the link is far too weak to decode anything:

  1. How much gain does the path actually have? (`--tx-gains` sweep)
  2. Is the receiver's own noise floor clean, or impulsive? (`--noise-only`)

It transmits an unmodulated tone rather than a burst. A tone has no PAPR
backoff, so its average power is ~10 dB above the burst's for the same DAC peak,
and the whole capture can be integrated coherently into one FFT bin. Together
that is tens of dB more sensitivity than preamble correlation -- which is the
difference between "0/10 bursts, no idea why" and a measured path gain.

The loopback backend is used as the reference model: the same measurement run
against it is, by definition, the design operating point, so the shortfall
column needs no hand-derived constant.

Usage:
    python -m hwlab.scripts.check_path --backend loopback
    python -m hwlab.scripts.check_path --backend hackrf
    python -m hwlab.scripts.check_path --backend hackrf --both-directions
    python -m hwlab.scripts.check_path --backend hackrf --noise-only

SAFETY: this sweeps TX gain up to whatever --tx-gains says (default max 40).
Keep the attenuator in the path, and change attenuation one step at a time,
re-measuring in between. The HackRF receiver is damaged above roughly -5 dBm.
"""
from __future__ import annotations

import argparse

import numpy as np

from hwlab.config import GainConfig, HwConfig
from hwlab.dsp import pulse
from hwlab.dsp.burst import BurstCodec
from hwlab.radio.loopback import LoopbackBackend
from hwlab.scripts._common import (
    add_common_args,
    build_backend,
    build_config,
    clock_pair,
    report_setup_problems,
)

#: Long enough that one FFT bin is a few Hz, so the coherent gain against noise
#: is large, while a capture still takes well under a second.
CAPTURE_SAMPLES = 262_144

#: The tone is searched for, not assumed, inside this window around the IF. A
#: clock-locked pair puts it exactly on the IF; an unlocked pair (which is the
#: normal state when --both-directions reverses the roles) offsets it by the
#: crystal difference, several kHz at 915 MHz.
SEARCH_HZ = 40e3


def make_tone(cfg: HwConfig, n: int) -> tuple[np.ndarray, float]:
    """Constant-envelope tone near the IF, at the same DAC peak a burst uses.

    The frequency is snapped to an exact whole number of cycles per `n` samples.
    Both backends transmit by repeating this block, and a block that does not
    close on a whole cycle puts a phase discontinuity at every repeat, which
    smears the tone and reads as several dB of path loss that is not there.
    The snap moves the tone by at most fs/2n -- tens of Hz.
    """
    cycles = max(1, round(cfg.link.if_offset_hz * n / cfg.link.fs))
    f = cycles * cfg.link.fs / n
    return cfg.link.dac_peak * np.exp(2j * np.pi * cycles * np.arange(n) / n), f


def find_tone(rx: np.ndarray, fs: float, f_center: float) -> tuple[float, float, bool]:
    """Locate the tone and return (amplitude_fullscale, offset_hz, detected).

    Amplitude is in the same normalized full-scale units as the transmitted
    tone, so 20*log10(rx_amp / tx_amp) is the end-to-end path gain.
    """
    n = len(rx)
    w = np.hanning(n)
    nfft = 1 << (int(np.ceil(np.log2(n))) + 2)  # zero-pad 4x: scalloping loss < 0.1 dB
    spec = np.fft.fftshift(np.fft.fft(rx * w, nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))

    # A complex tone of amplitude A lands as |X| = A * sum(w) in its bin.
    mag = np.abs(spec) / np.sum(w)
    near = np.abs(freqs - f_center) <= SEARCH_HZ
    peak = int(np.argmax(np.where(near, mag, 0.0)))

    far = np.abs(freqs - f_center) > 4 * SEARCH_HZ
    floor = float(np.median(mag[far])) if far.any() else 0.0
    return float(mag[peak]), float(freqs[peak] - f_center), mag[peak] > 6.0 * floor


def measure(backend, cfg: HwConfig, tone: np.ndarray, f_tone: float) -> tuple[float, float, bool, dict]:
    """One capture -> (path_gain_db, freq_error_hz, detected, rx stats)."""
    rx = backend.send_and_capture(tone, CAPTURE_SAMPLES)
    amp, foff, detected = find_tone(rx, cfg.link.fs, f_tone)
    gain_db = 20.0 * np.log10(max(amp, 1e-12) / cfg.link.dac_peak)
    return gain_db, foff, detected, rx_report(rx)


#: Above this, the pair is not sharing a reference. A burst is 8.71 ms long, so
#: even a few hundred Hz rotates it through whole cycles and no channel estimate
#: taken at the preamble is still valid at the data symbols.
UNLOCKED_HZ = 500.0


def clock_warning(freq_errors: list[float]) -> str:
    """Flag a frequency error that means the two radios are free-running."""
    worst = max((abs(f) for f in freq_errors), default=0.0)
    if worst <= UNLOCKED_HZ:
        return ""
    return (
        f"frequency error reaches {worst/1e3:.1f} kHz, so the pair is NOT sharing a reference. "
        f"Tone levels above are still valid -- the peak is searched for, not assumed -- but "
        f"bursts cannot be synchronized in this state."
    )


def trust_warnings(rep: dict) -> list[str]:
    """Reasons the numbers on this row should not be believed.

    An impulsive receiver puts narrowband spurs all over the search window, and
    the peak picker will happily lock onto one and report a confident path gain
    for a signal that is not ours. The frequency error column gives it away, so
    it is called out rather than silently averaged in.
    """
    out = []
    if rep["peak_lsb"] >= 120:
        out.append(f"receiver at full scale (peak {rep['peak_lsb']:.0f}/127 LSB)")
    if rep["peak_over_rms"] > 10.0:
        out.append(f"impulsive receiver (peak/rms {rep['peak_over_rms']:.0f}x)")
    return out


def rx_report(rx: np.ndarray) -> dict:
    comps = np.concatenate([rx.real, rx.imag]) * 127.0
    a = np.abs(comps)
    rms = float(np.sqrt(np.mean(comps**2)))
    return {
        "rms_lsb": rms,
        "peak_lsb": float(a.max()),
        "peak_over_rms": float(a.max() / rms) if rms > 0 else float("inf"),
        "p99_lsb": float(np.percentile(a, 99)),
        "p999_lsb": float(np.percentile(a, 99.9)),
        "clip_pct": pulse.clipping_fraction(rx) * 100.0,
        "dc_lsb": complex(np.mean(rx)) * 127.0,
    }


def reference_gain_db(cfg: HwConfig, tx_gain: float) -> float:
    """Path gain of the design model, measured exactly like the hardware is."""
    backend = LoopbackBackend(cfg.loopback)
    backend.configure(GainConfig(tx_vga_db=tx_gain, tx_amp=False,
                                 rx_lna_db=cfg.gains.rx_lna_db, rx_vga_db=cfg.gains.rx_vga_db))
    tone, f = make_tone(cfg, BurstCodec(cfg.link, cfg.burst).burst_samples)
    rx = backend.send_and_capture(tone, CAPTURE_SAMPLES)
    amp, _, _ = find_tone(rx, cfg.link.fs, f)
    return 20.0 * np.log10(max(amp, 1e-12) / cfg.link.dac_peak)


def sweep(backend, cfg: HwConfig, tx_gains: list[float], label: str) -> None:
    codec = BurstCodec(cfg.link, cfg.burst)
    tone, f_tone = make_tone(cfg, codec.burst_samples)

    print(f"\n== path gain: {label} ==")
    print("  (path gain = received tone / transmitted tone, both in full-scale units)")
    print(f"\n{'tx_vga':>7} {'path gain':>11} {'model':>9} {'short by':>9} "
          f"{'f err':>8} {'rx rms':>8} {'rx peak':>8} {'clip%':>7}")
    suspect, freq_errors = set(), []
    for tx_gain in tx_gains:
        backend.configure(GainConfig(tx_vga_db=tx_gain, tx_amp=False,
                                     rx_lna_db=cfg.gains.rx_lna_db, rx_vga_db=cfg.gains.rx_vga_db))
        gain_db, foff, detected, rep = measure(backend, cfg, tone, f_tone)
        model = reference_gain_db(cfg, tx_gain)
        suspect.update(trust_warnings(rep))
        if detected:
            freq_errors.append(foff)
        gain_s = f"{gain_db:>10.1f}" if detected else "  not seen"
        short_s = f"{model - gain_db:>8.1f}" if detected else "       --"
        foff_s = f"{foff/1e3:>7.1f}k" if detected else "       --"
        print(f"{tx_gain:7.0f} {gain_s} {model:9.1f} {short_s} {foff_s} "
              f"{rep['rms_lsb']:8.2f} {rep['peak_lsb']:8.0f} {rep['clip_pct']:7.4f}")

    for w in sorted(suspect):
        print(f"\n  WARNING: {w}. Treat this block's numbers as unreliable --")
        print("  the peak picker can lock onto a spur instead of the tone.")
    unlocked = clock_warning(freq_errors)
    if unlocked:
        print(f"\n  WARNING: {unlocked}")


def repeat(backend, cfg: HwConfig, n: int, tx_gain: float, label: str) -> None:
    """Measure the same point over and over. Is the path STABLE?

    Absolute level and frequency response both assume the bench holds still
    between captures. A connector that makes and breaks contact violates that,
    and then every other measurement is noise -- readings scatter by tens of dB
    and the operator chases a link budget that does not exist. This is the
    cheapest way to catch it, and it should be run before anything else is
    believed.

    The transmit-health column separates "the path dropped it" from "the
    transmitter never ran", which look identical at the receiver.
    """
    codec = BurstCodec(cfg.link, cfg.burst)
    tone, f_tone = make_tone(cfg, codec.burst_samples)
    backend.configure(GainConfig(tx_vga_db=tx_gain, tx_amp=False,
                                 rx_lna_db=cfg.gains.rx_lna_db, rx_vga_db=cfg.gains.rx_vga_db))

    print(f"\n== repeatability: {label} (tx_vga={tx_gain:.0f}, {n} runs) ==")
    print(f"\n{'#':>3} {'path gain':>11} {'f err':>8} {'rx rms':>8} {'rx peak':>8}  transmit")
    seen = []
    for i in range(n):
        gain_db, foff, detected, rep = measure(backend, cfg, tone, f_tone)
        if detected:
            seen.append(gain_db)
        gain_s = f"{gain_db:>10.1f}" if detected else "  not seen"
        foff_s = f"{foff/1e3:>7.1f}k" if detected else "       --"
        print(f"{i+1:3d} {gain_s} {foff_s} {rep['rms_lsb']:8.2f} {rep['peak_lsb']:8.0f}  "
              f"{backend.transmit_health()}")

    print(f"\n  detected     {len(seen)}/{n}")
    if seen:
        spread = max(seen) - min(seen)
        print(f"  path gain    median {float(np.median(seen)):.1f} dB, spread {spread:.1f} dB")
    if len(seen) < n:
        print("\n  INTERMITTENT. A live path delivers the tone on every run. Captures that fall")
        print("  back to the bare noise floor mean the signal stopped arriving, so a connector")
        print("  is making and breaking contact -- or, if the transmit column is unhealthy on")
        print("  those runs, the transmitter itself is dropping out.")
        print("  Nothing else measured on this bench is trustworthy until this is flat.")
    elif seen and max(seen) - min(seen) > 2.0:
        print("\n  UNSTABLE: a conducted path holds to well under 1 dB between captures.")
    else:
        print("\n  Stable. Absolute-level and frequency-response numbers can be believed.")


def freq_sweep(backend, cfg: HwConfig, freqs_hz: list[float], tx_gain: float, label: str) -> None:
    """Is the coupling conducted or radiated?

    A coax path with a fixed pad is flat across tens of MHz. Radiated leakage
    between two boards sitting on a bench goes through resonances and swings by
    many dB over the same span. That distinguishes "the cable is carrying the
    signal" from "the cable is dead and I am measuring the air" WITHOUT anyone
    having to unplug anything.
    """
    codec = BurstCodec(cfg.link, cfg.burst)
    tone, f_tone = make_tone(cfg, codec.burst_samples)
    backend.configure(GainConfig(tx_vga_db=tx_gain, tx_amp=False,
                                 rx_lna_db=cfg.gains.rx_lna_db, rx_vga_db=cfg.gains.rx_vga_db))
    original = cfg.link.center_freq_hz

    print(f"\n== frequency response: {label} (tx_vga={tx_gain:.0f}) ==")
    print(f"\n{'center MHz':>11} {'path gain':>11} {'f err':>8} {'rx rms':>8} {'rx peak':>8}")
    gains = []
    try:
        for f_hz in freqs_hz:
            cfg.link.center_freq_hz = f_hz
            gain_db, foff, detected, rep = measure(backend, cfg, tone, f_tone)
            gains.append(gain_db if detected else None)
            gain_s = f"{gain_db:>10.1f}" if detected else "  not seen"
            foff_s = f"{foff/1e3:>7.1f}k" if detected else "       --"
            print(f"{f_hz/1e6:11.1f} {gain_s} {foff_s} {rep['rms_lsb']:8.2f} {rep['peak_lsb']:8.0f}")
    finally:
        cfg.link.center_freq_hz = original

    seen = [g for g in gains if g is not None]
    if len(seen) < 2:
        print("\n  Too few detections to judge. Raise --freq-sweep-gain.")
        return
    spread = max(seen) - min(seen)
    print(f"\n  spread over {(max(freqs_hz)-min(freqs_hz))/1e6:.0f} MHz: {spread:.1f} dB")
    if spread <= 6.0:
        print("  Flat -- consistent with a CONDUCTED path. The coax is carrying the signal,")
        print("  so the shortfall is real loss in the cable/pad/connectors, not a dead cable.")
    else:
        print("  Swinging -- consistent with RADIATED leakage between the two radios rather")
        print("  than a conducted path. Check that the coax is seated on the ANT ports at")
        print("  both ends, and that the pad and cables are not open.")


#: Receive gain settings walked by --noise-only, from full gain down to none.
#: Analog noise must follow this; anything that does not is generated after the
#: gain stages and no gain setting will help.
_NOISE_GAINS = [(24.0, 20.0), (24.0, 8.0), (8.0, 8.0), (0.0, 0.0)]


def noise_verdict(rows: list[tuple[float, dict]]) -> tuple[str, float, float]:
    """Classify a receive-gain noise sweep as digital / impulsive / clean.

    `rows` is [(total_rx_gain_db, rx_report), ...] ordered from most gain to
    least. Returns (verdict, gain_span_db, rms_span_db).

    The digital test is deliberately generous -- noise is called gain-independent
    only if it follows less than a quarter of the gain change -- because the
    conclusion it licenses ("this radio cannot be used") is a strong one.
    """
    top, bottom = rows[0], rows[-1]
    gain_span = top[0] - bottom[0]
    rms_span = 20.0 * np.log10(top[1]["rms_lsb"] / max(bottom[1]["rms_lsb"], 1e-9))
    if gain_span > 0 and rms_span < 0.25 * gain_span and top[1]["peak_lsb"] >= 120:
        return "digital", gain_span, rms_span
    if top[1]["peak_over_rms"] > 10.0:
        return "impulsive", gain_span, rms_span
    return "clean", gain_span, rms_span


def noise_only(backend, cfg: HwConfig, label: str) -> None:
    """Receiver character with nothing transmitting, across receive gain.

    Two things are being separated here:

      thermal vs impulsive -- peak/rms. A complex Gaussian sits around 4-5x over
      this many samples; a small p99 beside a large p99.9 is impulsive.

      analog vs digital -- whether rms tracks the gain. Real noise entering the
      antenna port is amplified by the receive chain, so dropping the gain 44 dB
      must drop it 44 dB. Junk that stays put is being generated after the gain
      stages: USB transfer corruption, or the FPGA/ADC interface. That
      distinction decides whether the fix is a gain setting, a cable, or a
      different radio -- so it is measured rather than guessed.
    """
    print(f"\n== receiver noise: {label} ==")
    print("  transmitters idle\n")
    print(f"{'lna':>4} {'vga':>4} {'rms LSB':>9} {'peak':>6} {'pk/rms':>7} {'p99':>6} {'p99.9':>7} {'clip%':>8}")
    rows = []
    for lna, vga in _NOISE_GAINS:
        backend.configure(GainConfig(tx_vga_db=0.0, tx_amp=False, rx_lna_db=lna, rx_vga_db=vga))
        rep = rx_report(backend.capture_only(CAPTURE_SAMPLES))
        rows.append((lna + vga, rep))
        print(f"{lna:4.0f} {vga:4.0f} {rep['rms_lsb']:9.3f} {rep['peak_lsb']:6.0f} "
              f"{rep['peak_over_rms']:7.1f} {rep['p99_lsb']:6.1f} {rep['p999_lsb']:7.1f} "
              f"{rep['clip_pct']:8.4f}")

    verdict, gain_span, rms_span = noise_verdict(rows)
    print(f"\n  receive gain changed by {gain_span:.0f} dB; noise changed by {rms_span:+.1f} dB")

    if verdict == "digital":
        print("\n  DIGITAL, NOT ANALOG. Noise that ignores the receive gain is generated after")
        print("  the gain stages -- USB transfer corruption or the FPGA/ADC interface -- so no")
        print("  gain setting, attenuator, or antenna change will improve it.")
        print("  Try, in order: a different USB port (direct to the machine, not a hub), a")
        print("  different USB cable, then a different radio. Until this is clean, this unit")
        print("  cannot be used as the receiver: the glitches inflate the guard-region noise")
        print("  estimate and corrupt every SNR number measured through it.")
    elif verdict == "impulsive":
        print("\n  WARNING: peak/rms above 10x is impulsive interference, not thermal noise.")
        print("  It inflates the guard-region noise estimate and will read as a DSP fault.")
    else:
        print("\n  Clean: no full-scale glitches, and not impulsive.")
        if rms_span < 0.25 * gain_span:
            print("  Noise does not track the gain either, but it sits at a couple of LSB --")
            print("  that is the converter's own floor, i.e. the analog input is quieter than")
            print("  the ADC can resolve. Normal for a terminated port with no antenna.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--tx-gains", type=float, nargs="+", default=[0.0, 10.0, 20.0, 30.0, 40.0],
                        help="TX VGA settings to sweep, dB (0..47, whole dB)")
    parser.add_argument("--both-directions", action="store_true",
                        help="also measure with the transmit and receive units swapped. Equal results "
                             "point at the shared passive path; different results at one radio.")
    parser.add_argument("--noise-only", action="store_true",
                        help="characterize the receiver with nothing transmitting, and stop")
    parser.add_argument("--clock", choices=("leave", "on", "off"), default="leave",
                        help="drive the CLKOUT->CLKIN reference link before measuring. 'off' is a "
                             "diagnostic: if the slave's full-scale glitches vanish, the reference "
                             "link is corrupting its samples. Measurements taken with it off carry "
                             "a large carrier frequency offset and cannot be used for bursts.")
    parser.add_argument("--repeat", type=int, default=0, metavar="N",
                        help="measure the same point N times and report the scatter. Run this FIRST: "
                             "an intermittent connector makes every other number meaningless.")
    parser.add_argument("--repeat-gain", type=float, default=30.0,
                        help="TX VGA to hold during --repeat, dB")
    parser.add_argument("--freq-sweep", action="store_true",
                        help="measure path gain across center frequency instead of TX gain. Flat means "
                             "a conducted path; swinging means you are measuring leakage through the air.")
    parser.add_argument("--freq-grid", type=float, nargs="+",
                        default=[895e6, 905e6, 915e6, 925e6, 935e6],
                        help="center frequencies in Hz for --freq-sweep")
    parser.add_argument("--freq-sweep-gain", type=float, default=40.0,
                        help="TX VGA to hold during --freq-sweep, dB")
    args = parser.parse_args()
    return report_setup_problems(_run, args)


def _set_clock(args, cfg: HwConfig) -> None:
    if args.backend == "loopback" or args.clock == "leave":
        return
    from hwlab.radio.clock import check_clkin, disable_clkout, enable_clkout

    master, slave = clock_pair(cfg)
    act = enable_clkout if args.clock == "on" else disable_clkout
    act(master, cfg.device.hackrf_clock)
    detected, _ = check_clkin(slave, cfg.device.hackrf_clock)
    print(f"== clock ==\n  CLKOUT {args.clock} on …{master[-8:]}; "
          f"…{slave[-8:]} reports CLKIN {'detected' if detected else 'absent'}\n")


def _run(args) -> int:
    cfg = build_config(args)
    _set_clock(args, cfg)
    directions = [(cfg.device.tx_serial, cfg.device.rx_serial, "configured direction")]
    if args.both_directions:
        directions.append((cfg.device.rx_serial, cfg.device.tx_serial, "reversed"))

    for tx_serial, rx_serial, label in directions:
        cfg.device.tx_serial, cfg.device.rx_serial = tx_serial, rx_serial
        with build_backend(args, cfg) as backend:
            name = f"{label} (rx …{rx_serial[-8:]})" if rx_serial else label
            if args.noise_only:
                noise_only(backend, cfg, name)
            elif args.repeat:
                repeat(backend, cfg, args.repeat, args.repeat_gain, name)
            elif args.freq_sweep:
                freq_sweep(backend, cfg, args.freq_grid, args.freq_sweep_gain, name)
            else:
                sweep(backend, cfg, args.tx_gains, name)

    if not (args.noise_only or args.repeat or args.freq_sweep):
        print("\nReading this table:")
        print("  'short by' near 0        -- the path matches the design model; run check_link")
        print("  'short by' large but flat with tx_vga, or 'not seen' throughout")
        print("                           -- the coax is not carrying the signal. Disconnect it:")
        print("                              if nothing changes, what you are measuring is leakage")
        print("                              between the two radios, not the cable.")
        print("  a 10 dB attenuator swap that does NOT move path gain by 10 dB")
        print("                           -- a connector is not making contact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
