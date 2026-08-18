"""CLKOUT/CLKIN helpers -- the single most important piece of setup.

Wiring #1's CLKOUT (10 MHz, 3.3 V square wave) to #2's CLKIN with a short SMA
cable makes the pair coherent: both LOs and both sample clocks derive from one
reference, so the carrier frequency offset is zero rather than the tens of ppm
two free-running crystals would give. That is what makes an 8-symbol payload
synchronizable at all -- there is no room to estimate and track a frequency
offset inside 8 symbols.

Note: a HackRF only switches over to CLKIN when a TX or RX operation begins, so
`check_clkin` may need a transfer to have been started at least once.

The exact flags differ slightly between hackrf-tools releases; run
`hackrf_clock -h` if these fail and adjust in DeviceConfig.
"""
from __future__ import annotations

import shutil
import subprocess


class ClockError(RuntimeError):
    pass


def _run(exe: str, args: list[str], timeout: float = 10.0) -> str:
    if shutil.which(exe) is None:
        raise ClockError(
            f"'{exe}' not found on PATH. Install the hackrf tools (Windows: the "
            f"official hackrf release zip, then add its bin/ directory to PATH)."
        )
    proc = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    return (proc.stdout or "") + (proc.stderr or "")


def _device_args(serial: str) -> list[str]:
    return ["-d", serial] if serial else []


def enable_clkout(serial: str = "", exe: str = "hackrf_clock") -> str:
    """Turn on CLKOUT on the master unit."""
    return _run(exe, [*_device_args(serial), "-o", "1"])


def check_clkin(serial: str = "", exe: str = "hackrf_clock") -> tuple[bool, str]:
    """Ask the slave unit whether it sees an external reference.

    Returns (detected, raw_output). Look for 'clock signal detected'.
    """
    out = _run(exe, [*_device_args(serial), "-i"])
    return ("detected" in out.lower() and "not detected" not in out.lower()), out


def disable_clkout(serial: str = "", exe: str = "hackrf_clock") -> str:
    """Turn CLKOUT off. Used to prove whether the reference link is what is
    corrupting the slave's samples -- a marginal CLKIN can make an otherwise
    healthy radio produce full-scale glitches that no gain setting removes."""
    return _run(exe, [*_device_args(serial), "-o", "0"])


def require_locked_pair(master_serial: str, slave_serial: str, exe: str = "hackrf_clock") -> str:
    """Enable CLKOUT on the master, verify CLKIN on the slave, or raise.

    Master/slave is set by which way the CLOCK cable runs, which is independent
    of who transmits -- see DeviceConfig.clock_master_serial.
    """
    report = [enable_clkout(master_serial, exe)]
    detected, out = check_clkin(slave_serial, exe)
    report.append(out)
    if not detected:
        raise ClockError(
            "CLKIN not detected on the slave HackRF.\n"
            "Check: (1) SMA cable from the master's CLKOUT to the slave's CLKIN, "
            "(2) device.clock_master_serial names the unit the cable leaves from, "
            "(3) CLKIN is only sampled when a transfer starts -- try again "
            "after one transfer.\n"
            "Without a locked pair the carrier frequency offset will not be zero "
            "and an 8-symbol payload cannot be synchronized.\n\n" + "\n".join(report)
        )
    return "\n".join(report)
