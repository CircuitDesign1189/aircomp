"""SDRBackend: the seam between DSP and hardware.

The interface is deliberately narrow -- configure gains, then push a burst and
get a capture back -- so that LoopbackBackend and HackRFCLIBackend are truly
interchangeable and every script can be dry-run without a radio attached.

IQ convention on this interface: complex float64, normalized full scale, where
+-1.0 corresponds to +-127 LSB of the 8-bit converters. Backends perform their
own int8 conversion so quantization is modeled identically on both sides.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from hwlab.config import GainConfig


class SDRBackend(ABC):
    @abstractmethod
    def configure(self, gains: GainConfig) -> None:
        """Set transmit and receive gains. Called before send_and_capture."""

    def preflight(self) -> str:  # pragma: no cover - trivial
        """Check the hardware is usable before anything transmits.

        Returns a one-line report. Backends without hardware say so and do nothing.
        """
        return "(no hardware to check)"

    @abstractmethod
    def send_and_capture(self, tx_iq: np.ndarray, capture_samples: int) -> np.ndarray:
        """Transmit `tx_iq` (repeating) while capturing `capture_samples` samples.

        Repeat-transmit plus an over-long capture is what removes the need for
        hardware triggering: the capture is guaranteed to contain at least one
        complete burst regardless of process start-up jitter, and preamble
        correlation finds it.
        """

    @abstractmethod
    def capture_only(self, capture_samples: int) -> np.ndarray:
        """Receive with nothing transmitting -- the receiver's own noise floor.

        Separating this from send_and_capture is what makes "is the receiver
        quiet?" answerable independently of "is the link working?".
        """

    def transmit_health(self) -> str:  # pragma: no cover - trivial
        """One line on whether the last transmit actually radiated.

        Needed to tell "the path dropped the signal" apart from "the transmitter
        never ran", which look identical from the receiver.
        """
        return "n/a"

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
