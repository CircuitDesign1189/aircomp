# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Burst layout: guard | preamble | pilots | data | guard.

Design notes:

* The payload is tiny -- k=16 reals is 8 complex symbols per message -- so
  sync/pilot overhead dominates by design. That cost is real and is reported
  honestly by `BurstLayout.overhead_symbols`; see CLAUDE.md's note that bits
  vs symbols is not an apples-to-apples comparison.

* The preamble is a Zadoff-Chu sequence: sharp autocorrelation AND constant
  envelope. Constant envelope matters here because an 8-bit DAC has little
  headroom to waste on peaks. Its length buys correlation processing gain,
  which is what lets frame detection survive at the bottom of the sweep --
  see BurstConfig for the measured loss rates that set the default.

* Every reference symbol is scaled to |s|^2 = 2, matching the average power of
  a data symbol under the convention in `hwlab/dsp/mapping.py`. This is what
  makes the pilot-derived channel estimate directly applicable to the data
  symbols without a correction factor.

* The guard regions are transmitted as zeros and exist to measure the noise
  floor. They are the reason we can label the x-axis with *measured* SNR
  rather than with nominal gain settings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: |s|^2 for every reference (preamble/pilot) symbol, matching E|data|^2 = 2.
REF_SYMBOL_POWER = 2.0


def zadoff_chu(length: int, root: int = 25) -> np.ndarray:
    """Constant-envelope ZC sequence, scaled to REF_SYMBOL_POWER."""
    if length % 2 == 0:
        raise ValueError("use an odd Zadoff-Chu length for the closed-form below")
    if np.gcd(root, length) != 1:
        raise ValueError(f"ZC root {root} must be coprime with length {length}")
    n = np.arange(length)
    seq = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    return seq * np.sqrt(REF_SYMBOL_POWER)


def pilot_sequence(count: int, seed: int = 0xA1C0) -> np.ndarray:
    """Deterministic QPSK pilots at |s|^2 = REF_SYMBOL_POWER."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(count, 2)) * 2 - 1
    return (bits[:, 0] + 1j * bits[:, 1]).astype(complex)  # |s|^2 = 2 already


@dataclass(frozen=True)
class BurstLayout:
    guard_symbols: int = 160
    preamble_len: int = 511  # see BurstConfig for the sync-reliability measurements
    zc_root: int = 25
    n_pilots: int = 32
    n_data: int = 8  # k=16 reals -> 8 complex symbols

    @property
    def preamble_start(self) -> int:
        return self.guard_symbols

    @property
    def pilot_start(self) -> int:
        return self.preamble_start + self.preamble_len

    @property
    def data_start(self) -> int:
        return self.pilot_start + self.n_pilots

    @property
    def total_symbols(self) -> int:
        return self.data_start + self.n_data + self.guard_symbols

    @property
    def overhead_symbols(self) -> int:
        """Non-payload symbols per burst. Report this; do not quietly drop it."""
        return self.total_symbols - self.n_data

    def preamble(self) -> np.ndarray:
        return zadoff_chu(self.preamble_len, self.zc_root)

    def pilots(self) -> np.ndarray:
        return pilot_sequence(self.n_pilots)

    def build(self, data_symbols: np.ndarray) -> np.ndarray:
        """Assemble the full symbol sequence for one burst."""
        data_symbols = np.asarray(data_symbols, dtype=complex)
        if len(data_symbols) != self.n_data:
            raise ValueError(f"expected {self.n_data} data symbols, got {len(data_symbols)}")
        burst = np.zeros(self.total_symbols, dtype=complex)
        burst[self.preamble_start : self.pilot_start] = self.preamble()
        burst[self.pilot_start : self.data_start] = self.pilots()
        burst[self.data_start : self.data_start + self.n_data] = data_symbols
        return burst

    def guard_slice(self, margin: int = 8) -> slice:
        """Interior of the leading guard, clear of filter transients and of the
        tail of a preceding burst when the transmitter is in repeat mode."""
        stop = self.guard_symbols - margin
        start = margin
        if stop - start < 32:
            raise ValueError("guard_symbols too small for a reliable noise estimate")
        return slice(start, stop)
