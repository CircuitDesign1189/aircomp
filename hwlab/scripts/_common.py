"""Shared CLI plumbing for the hwlab scripts."""
from __future__ import annotations

import argparse

from typing import Callable

from hwlab.config import HwConfig, load_hw_config
from hwlab.radio.backend import SDRBackend
from hwlab.radio.clock import ClockError
from hwlab.radio.hackrf_cli import HackRFCLIBackend, HackRFError
from hwlab.radio.loopback import LoopbackBackend


def report_setup_problems(run: Callable[[object], int], args) -> int:
    """Run a script body, turning bench faults into a message and an exit code.

    A missing cable, a stranded radio, or an unlocked clock pair is the expected
    outcome of a hardware script, not a bug in it -- a traceback buries the one
    line the operator needs.
    """
    try:
        return run(args)
    except (HackRFError, ClockError) as exc:
        print(f"\nSETUP PROBLEM\n\n{exc}")
        return 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="hwlab/configs/sdr_link.yaml")
    parser.add_argument(
        "--backend",
        choices=("loopback", "hackrf"),
        default="loopback",
        help="'loopback' runs the full DSP chain in numpy with no radio attached -- "
        "use it to validate everything before touching hardware.",
    )
    parser.add_argument("--tx-serial", default=None, help="override device.tx_serial")
    parser.add_argument("--rx-serial", default=None, help="override device.rx_serial")


def build_config(args) -> HwConfig:
    cfg = load_hw_config(args.config)
    if getattr(args, "tx_serial", None):
        cfg.device.tx_serial = args.tx_serial
    if getattr(args, "rx_serial", None):
        cfg.device.rx_serial = args.rx_serial
    return cfg


def build_backend(args, cfg: HwConfig) -> SDRBackend:
    """Build the backend and prove the hardware is usable before anything transmits.

    The preflight runs here rather than inside the backend constructor so the
    constructor stays cheap and testable without a radio.
    """
    if args.backend == "loopback":
        return LoopbackBackend(cfg.loopback)
    backend = HackRFCLIBackend(cfg.link, cfg.device)
    try:
        print(f"== devices ==\n  {backend.preflight()}\n")
    except BaseException:
        backend.close()
        raise
    return backend


def clock_pair(cfg: HwConfig) -> tuple[str, str]:
    """(master, slave) for the reference link, from the CABLE not the roles."""
    master = cfg.device.clock_master_serial or cfg.device.tx_serial
    slave = cfg.device.rx_serial if master == cfg.device.tx_serial else cfg.device.tx_serial
    return master, slave


def check_clock(args, cfg: HwConfig) -> str:
    """Verify the CLKOUT -> CLKIN reference link. No-op on loopback."""
    if args.backend == "loopback":
        return "(loopback: no clock to check)"
    from hwlab.radio.clock import require_locked_pair

    master, slave = clock_pair(cfg)
    return require_locked_pair(master, slave, cfg.device.hackrf_clock)
