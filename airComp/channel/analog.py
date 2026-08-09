"""Differentiable analog AWGN channel used by the semantic/JSCC pipeline.

`z` is assumed power-normalized upstream (see SemanticEncoder), so `SNR_dB`
directly parameterizes the noise variance against unit average signal power.
Implemented as a plain tensor op so gradients flow through it end-to-end
(the standard "deep JSCC" trick, e.g. Bourtsoulatze et al. 2019).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnalogAWGNChannel(nn.Module):
    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        snr_linear = 10 ** (snr_db / 10.0)
        sigma = (1.0 / snr_linear) ** 0.5
        noise = torch.randn_like(z) * sigma
        return z + noise
