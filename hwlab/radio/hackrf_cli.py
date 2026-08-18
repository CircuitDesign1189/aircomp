"""HackRF backend driven by the official `hackrf_transfer` command-line tool.

Why subprocess rather than a Python binding: `hackrf_transfer` ships with the
official hackrf tools and works on Windows out of the box, whereas SoapySDR and
python_hackrf both need a build toolchain and are a common place to lose a day.
The cost is roughly 1-2 s per burst, which is nothing next to the LLM
generation that dominates each negotiation turn. If that ever matters, a
python_hackrf backend can be dropped in behind the same SDRBackend interface.

The burst is repeated back-to-back inside the transmit FILE, long enough to
cover the whole capture, so the capture is guaranteed to contain at least one
complete burst regardless of process start-up jitter. That is what removes the
need for a hardware trigger.

Why a finite file rather than `hackrf_transfer -R` (infinite repeat): a repeat
transmitter never ends by itself, so it has to be killed. On Windows
`Popen.terminate()` is `TerminateProcess` (and `Popen.kill` is an *alias* for
it, so there is no stronger escalation), which skips libhackrf's
`hackrf_stop_tx()`/`hackrf_close()`. If the process is inside an uninterruptible
WinUSB bulk transfer when that lands, it never finishes dying: it sits in a
terminating state holding both the transmit file and the USB interface, and
every later `hackrf_transfer` on that unit fails with
`hackrf_open() failed: HackRF not found (-5)` until the radio is physically
replugged. A finite file lets the tool reach its own exit path instead.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from hwlab.config import DeviceConfig, GainConfig, LinkConfig
from hwlab.dsp import pulse
from hwlab.radio.backend import SDRBackend


class HackRFError(RuntimeError):
    pass


_STUCK_UNIT_HINT = (
    "A HackRF is attached but cannot be opened, which means another process still holds "
    "it. Check for a stranded transmitter:\n"
    "    Get-CimInstance Win32_Process -Filter \"Name='hackrf_transfer.exe'\"\n"
    "If one is listed, `taskkill /F` will not necessarily clear it -- a process wedged in "
    "a USB transfer is only released by unplugging and replugging that unit. Confirm with "
    "`hackrf_info` that both serials are listed before retrying."
)


class HackRFCLIBackend(SDRBackend):
    def __init__(
        self,
        link: LinkConfig,
        device: DeviceConfig | None = None,
        discard_samples: int = 200_000,  # 0.1 s at 2 Msps: PLL settling + AGC-free startup transient
        timeout_s: float = 60.0,
    ):
        self.link = link
        self.device = device or DeviceConfig()
        self.gains = GainConfig()
        # Transmit timing lives in DeviceConfig: it is a property of the host and its USB
        # stack, not of the waveform, and tx_margin_s has to be reachable from the config
        # because it is what the "transmitter ran out of samples" error asks you to raise.
        self.tx_settle_s = self.device.tx_settle_s
        self.tx_margin_s = self.device.tx_margin_s
        self.discard_samples = discard_samples
        self.timeout_s = timeout_s

        if shutil.which(self.device.hackrf_transfer) is None:
            raise HackRFError(
                f"'{self.device.hackrf_transfer}' not found on PATH. Install the hackrf "
                f"tools, or use LoopbackBackend to develop without hardware."
            )
        self._tmp = None
        if self.device.workdir:
            self.workdir = Path(self.device.workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
        else:
            # ignore_cleanup_errors: a stranded transmitter can still hold tx.bin, and a
            # failure to delete a temp file must never replace the error that caused it.
            self._tmp = tempfile.TemporaryDirectory(prefix="hwlab_", ignore_cleanup_errors=True)
            self.workdir = Path(self._tmp.name)

    # -- SDRBackend -----------------------------------------------------

    def preflight(self) -> str:
        """Fail before transmitting if a configured unit is missing or already claimed.

        Without this the first symptom of a stranded unit is `HackRF not found (-5)`
        from a burst that is already half set up, which reads like a wiring fault
        rather than a stuck process.
        """
        units = probe_devices(self.device.hackrf_info)
        if not units:
            raise HackRFError(
                f"no HackRF detected by '{self.device.hackrf_info}'. Check USB, or use "
                f"--backend loopback to work without hardware."
            )

        serials = [s for _, s in units if s]
        unopenable = [i for i, s in units if not s]
        wanted = {"tx_serial": self.device.tx_serial, "rx_serial": self.device.rx_serial}

        if self.device.tx_serial and self.device.tx_serial == self.device.rx_serial:
            raise HackRFError(
                "tx_serial and rx_serial are the same unit. The HackRF One is half-duplex "
                "and this bench transmits and receives at the same time, so two units are "
                "required."
            )
        if not self.device.tx_serial and not self.device.rx_serial and len(units) > 1:
            raise HackRFError(
                f"{len(units)} HackRFs are attached but device.tx_serial/rx_serial are empty, "
                f"so both transmit and receive would open the same unit. Fill them in from "
                f"`hackrf_info` (attached: {', '.join(serials) or 'none readable'})."
            )

        for field, serial in wanted.items():
            if serial and serial.lower() not in [s.lower() for s in serials]:
                if unopenable:
                    raise HackRFError(
                        f"{field}={serial} is not among the readable units "
                        f"(index {', '.join(str(i) for i in unopenable)} could not be opened).\n"
                        + _STUCK_UNIT_HINT
                    )
                raise HackRFError(
                    f"{field}={serial} is not attached. Units found: {', '.join(serials)}."
                )
        if unopenable:
            raise HackRFError(
                f"HackRF index {', '.join(str(i) for i in unopenable)} could not be opened.\n"
                + _STUCK_UNIT_HINT
            )
        return f"{len(units)} HackRF(s) available: {', '.join(serials)}"

    def configure(self, gains: GainConfig) -> None:
        _check_gain_steps(gains)
        self.gains = gains

    def send_and_capture(self, tx_iq: np.ndarray, capture_samples: int) -> np.ndarray:
        tx_path = self.workdir / "tx.bin"
        rx_path = self.workdir / "rx.bin"
        log_path = self.workdir / "tx_log.txt"

        total = capture_samples + self.discard_samples
        tx_samples = self._write_tx_file(tx_iq, total, tx_path)

        # stdout goes to a file, not a pipe: nothing drains a pipe while the receiver
        # runs, so a chatty transmitter could otherwise fill the buffer and block.
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            tx_proc = subprocess.Popen(
                self._tx_cmd(tx_path, tx_samples), stdout=log, stderr=subprocess.STDOUT, text=True
            )
            try:
                time.sleep(self.tx_settle_s)
                if tx_proc.poll() is not None:
                    raise HackRFError(
                        "transmit process exited early:\n" + _read_log(log_path) + "\n" + _STUCK_UNIT_HINT
                    )
                rx = subprocess.run(
                    self._rx_cmd(rx_path, total),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                if rx.returncode != 0:
                    raise HackRFError(
                        f"receive failed (exit {rx.returncode}):\n{rx.stdout}\n{rx.stderr}\n"
                        + _STUCK_UNIT_HINT
                    )
                tx_ended_early = tx_proc.poll() is not None
            finally:
                # No raising in the finally: it must not replace an in-flight exception.
                stop_error = self._stop_tx(tx_proc, tx_samples)

        if stop_error:
            raise HackRFError(stop_error)
        if tx_ended_early:
            raise HackRFError(
                f"the transmitter ran out of samples before the capture finished, so the "
                f"tail of the capture is silence. Increase tx_margin_s (currently "
                f"{self.tx_margin_s} s)."
            )

        return self._read_capture(rx_path, total)

    def capture_only(self, capture_samples: int) -> np.ndarray:
        total = capture_samples + self.discard_samples
        rx_path = self.workdir / "rx.bin"
        rx = subprocess.run(
            self._rx_cmd(rx_path, total), capture_output=True, text=True, timeout=self.timeout_s
        )
        if rx.returncode != 0:
            raise HackRFError(
                f"receive failed (exit {rx.returncode}):\n{rx.stdout}\n{rx.stderr}\n" + _STUCK_UNIT_HINT
            )
        return self._read_capture(rx_path, total)

    def transmit_health(self) -> str:
        """What hackrf_transfer reported about the last transmit.

        'average power' is the level the tool actually fed to the DAC, so a
        healthy line here rules the transmitter out and points at the RF path.
        """
        log = _read_log(self.workdir / "tx_log.txt")
        power = _POWER_RE.search(log)
        if power:
            clean = "hackrf_close() done" in log
            return f"{power.group(1)} dBfs" + ("" if clean else " (no clean shutdown!)")
        if not log:
            return "no log"
        return "NO POWER LINE -- transmitter did not run"

    def _read_capture(self, rx_path: Path, total: int) -> np.ndarray:
        iq = pulse.dequantize_int8(np.fromfile(rx_path, dtype=np.int8))
        if len(iq) < total:
            raise HackRFError(f"short capture: got {len(iq)} of {total} samples")
        return iq[self.discard_samples : total]

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()  # ignore_cleanup_errors=True, so this cannot raise
            self._tmp = None

    # -- transmit lifetime ------------------------------------------------

    def _write_tx_file(self, tx_iq: np.ndarray, total: int, path: Path) -> int:
        """Write whole bursts back-to-back, enough to outlast the capture.

        Returns the sample count actually written. Sizing it here -- rather than
        transmitting forever and killing the process -- is what keeps the radio
        recoverable; see the module docstring.
        """
        burst = np.asarray(tx_iq, dtype=complex)
        if len(burst) == 0:
            raise HackRFError("tx_iq is empty")
        wanted = int(self.link.fs * (self.tx_settle_s + self.tx_margin_s)) + total
        reps = max(1, -(-wanted // len(burst)))  # ceil, so a whole burst is never cut
        np.tile(pulse.quantize_int8(burst), reps).tofile(path)
        return reps * len(burst)

    def _stop_tx(self, tx_proc: subprocess.Popen, tx_samples: int) -> str:
        """Wait for the finite transmit to end on its own. Returns "" or an error."""
        grace = tx_samples / self.link.fs + 5.0
        try:
            tx_proc.wait(timeout=grace)
            return ""
        except subprocess.TimeoutExpired:  # pragma: no cover - hardware only
            pass
        # Should be unreachable: the transmit file is finite. If it happens, killing is
        # the only option left, and it is exactly the operation that can strand the unit.
        tx_proc.terminate()
        try:
            tx_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return (
            f"the transmitter did not finish within {grace:.1f} s and had to be killed. "
            f"The unit may now be stranded.\n" + _STUCK_UNIT_HINT
        )

    # -- command construction (kept separate so it is unit-testable) -----

    def _tx_cmd(self, path: Path, num_samples: int) -> list[str]:
        cmd = [self.device.hackrf_transfer]
        if self.device.tx_serial:
            cmd += ["-d", self.device.tx_serial]
        cmd += [
            "-t", str(path),
            "-f", str(int(self.link.center_freq_hz)),
            "-s", str(int(self.link.fs)),
            "-x", str(int(round(self.gains.tx_vga_db))),
            "-a", "1" if self.gains.tx_amp else "0",
            # Belt and braces with the finite file: whichever limit hackrf_transfer
            # honours first, it reaches its own clean shutdown. No -R, ever.
            "-n", str(int(num_samples)),
        ]
        return cmd

    def _rx_cmd(self, path: Path, num_samples: int) -> list[str]:
        cmd = [self.device.hackrf_transfer]
        if self.device.rx_serial:
            cmd += ["-d", self.device.rx_serial]
        cmd += [
            "-r", str(path),
            "-f", str(int(self.link.center_freq_hz)),
            "-s", str(int(self.link.fs)),
            "-l", str(int(round(self.gains.rx_lna_db))),
            "-g", str(int(round(self.gains.rx_vga_db))),
            "-a", "0",  # RX RF amp off: SNR is controlled by TX gain and the attenuator only
            "-n", str(int(num_samples)),
        ]
        return cmd


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:  # pragma: no cover - defensive
        return "(transmit log unavailable)"


_POWER_RE = re.compile(r"average power ([-\d.]+) dBfs")
_INDEX_RE = re.compile(r"^Index:\s*(\d+)", re.MULTILINE)
_SERIAL_RE = re.compile(r"^Serial number:\s*([0-9a-fA-F]+)", re.MULTILINE)


def probe_devices(exe: str = "hackrf_info") -> list[tuple[int, str]]:
    """Enumerate attached HackRFs as (index, serial) pairs.

    A unit that is present but cannot be opened -- typically because another
    process still holds it -- is reported with an empty serial, because that is
    exactly how `hackrf_info` reports it: the "Found HackRF / Index: n" block
    appears, but the details below it do not.
    """
    if shutil.which(exe) is None:
        raise HackRFError(
            f"'{exe}' not found on PATH. Install the hackrf tools, or use "
            f"LoopbackBackend to develop without hardware."
        )
    try:
        proc = subprocess.run([exe], capture_output=True, text=True, timeout=20.0)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - hardware only
        raise HackRFError(f"'{exe}' did not respond within 20 s.\n" + _STUCK_UNIT_HINT) from exc

    units: list[tuple[int, str]] = []
    for block in ((proc.stdout or "") + "\n" + (proc.stderr or "")).split("Found HackRF")[1:]:
        index = _INDEX_RE.search(block)
        serial = _SERIAL_RE.search(block)
        if index:
            units.append((int(index.group(1)), serial.group(1) if serial else ""))
    return units


def _check_gain_steps(gains: GainConfig) -> None:
    """The HackRF quantizes gains to hardware steps; asking for 25 dB of LNA
    silently gets you 24. Fail loudly instead, so a calibration table never
    records a gain the radio did not actually apply."""
    problems = []
    if not 0 <= gains.tx_vga_db <= 47:
        problems.append(f"tx_vga_db={gains.tx_vga_db} outside 0..47")
    if gains.tx_vga_db % 1 != 0:
        problems.append(f"tx_vga_db={gains.tx_vga_db} is not a whole dB (1 dB steps)")
    if not 0 <= gains.rx_lna_db <= 40 or gains.rx_lna_db % 8 != 0:
        problems.append(f"rx_lna_db={gains.rx_lna_db} must be 0..40 in 8 dB steps")
    if not 0 <= gains.rx_vga_db <= 62 or gains.rx_vga_db % 2 != 0:
        problems.append(f"rx_vga_db={gains.rx_vga_db} must be 0..62 in 2 dB steps")
    if problems:
        raise HackRFError("invalid HackRF gain settings: " + "; ".join(problems))
