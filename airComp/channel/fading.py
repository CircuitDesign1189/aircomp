"""Optional Rayleigh block-fading channel (stretch goal, not used by the
default Milestone 1/2 pipelines). Applies one fading magnitude per
transmission, then AWGN, then perfect-CSI equalization at the receiver.

Symbols here are real-valued (not complex I/Q), so this uses the magnitude
of a 2D Gaussian as a real-valued stand-in for Rayleigh-distributed fading.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from airComp.channel.analog import AnalogAWGNChannel


class RayleighBlockFadingChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.awgn = AnalogAWGNChannel()

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        h_real = torch.randn(1, device=z.device)
        h_imag = torch.randn(1, device=z.device)
        h_mag = torch.sqrt(h_real**2 + h_imag**2) / (2**0.5)
        faded = z * h_mag
        received = self.awgn(faded, snr_db)
        return received / h_mag.clamp_min(1e-6)
