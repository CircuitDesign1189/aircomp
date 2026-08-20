# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""LoopbackBackend -- the whole hardware link, in numpy.

This is not a stub. It applies the same impairments the real link has, in the
same order, so the full DSP chain (framing, IF, sync, LS/ZF equalization, noise
measurement, 8-bit quantization) is exercised and unit-tested before any radio
is powered on.

Noise is injected at the *RX input*, i.e. ahead of the receive gain, so RX gain
amplifies signal and noise together and leaves SNR unchanged -- as it does in
reality. RX gain therefore only affects ADC headroom, which is exactly the
lesson the operator needs: move SNR with TX gain and the attenuator, never with
RX gain.
"""
from __future__ import annotations

import numpy as np

from hwlab.config import GainConfig, LoopbackConfig
from hwlab.dsp import pulse
from hwlab.radio.backend import SDRBackend


class LoopbackBackend(SDRBackend):
    def __init__(self, cfg: LoopbackConfig | None = None):
        self.cfg = cfg or LoopbackConfig()
        self.gains = GainConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def configure(self, gains: GainConfig) -> None:
        self.gains = gains

    def _tx_amplitude(self) -> float:
        db = self.gains.tx_vga_db + (14.0 if self.gains.tx_amp else 0.0) - self.cfg.path_loss_db
        return 10.0 ** (db / 20.0)

    def _rx_amplitude(self) -> float:
        return 10.0 ** ((self.gains.rx_lna_db + self.gains.rx_vga_db) / 20.0)

    def _apply_iq_imbalance(self, x: np.ndarray) -> np.ndarray:
        g = self.cfg.iq_gain_imbalance
        phi = np.deg2rad(self.cfg.iq_phase_deg)
        if g == 0.0 and phi == 0.0:
            return x
        i = x.real * (1.0 + g / 2.0)
        q = x.imag * (1.0 - g / 2.0)
        return (i + q * np.sin(phi)) + 1j * (q * np.cos(phi))

    def capture_only(self, capture_samples: int) -> np.ndarray:
        return self._receive(np.zeros(capture_samples, dtype=complex))

    def send_and_capture(self, tx_iq: np.ndarray, capture_samples: int) -> np.ndarray:
        tx = np.asarray(tx_iq, dtype=complex)
        if self.cfg.quantize:
            tx = pulse.dequantize_int8(pulse.quantize_int8(tx))  # DAC

        offset = int(self.cfg.timing_offset_samples) % max(len(tx), 1)
        reps = int(np.ceil((capture_samples + offset) / len(tx))) + 1
        stream = np.tile(tx, reps)[offset : offset + capture_samples]

        signal = stream * self._tx_amplitude() * np.exp(1j * np.deg2rad(self.cfg.phase_offset_deg))
        return self._receive(signal)

    def _receive(self, signal: np.ndarray) -> np.ndarray:
        """Add the RX-input noise, apply receive gain, and quantize."""
        n = len(signal)
        sigma = 10.0 ** (self.cfg.noise_floor_dbfs / 20.0)  # per-real STD (dbfs is a power figure)
        noise = self.rng.normal(0.0, sigma, n) + 1j * self.rng.normal(0.0, sigma, n)

        rx = (signal + noise) * self._rx_amplitude() + self.cfg.dc_offset
        rx = self._apply_iq_imbalance(rx)

        if self.cfg.quantize:
            rx = pulse.dequantize_int8(pulse.quantize_int8(rx))  # ADC (clips at full scale)
        return rx
