"""SDRAnalogChannel -- a real RF link with the same call signature as
`airComp.channel.analog.AnalogAWGNChannel`.

    y = channel(z, snr_db)

`SemanticAgent.take_turn` runs under `@torch.no_grad()`, so no gradients are
needed and this can be a plain callable rather than an nn.Module. Training
still uses the differentiable simulation channel -- `analog.py` is not touched.

`snr_db` is a *request*, not a command: the hardware delivers whatever the
calibrated gain setting actually produces, and the achieved value is recorded
in `stats_log[-1]["measured_snr_db"]`. Plot against that, not against the
request.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from hwlab.channel.calibration import Calibration
from hwlab.config import BurstConfig, GainConfig, LinkConfig
from hwlab.dsp.burst import BurstCodec
from hwlab.dsp.mapping import noise_var_per_real
from hwlab.radio.backend import SDRBackend


class SDRAnalogChannel:
    def __init__(
        self,
        backend: SDRBackend,
        link: LinkConfig | None = None,
        burst: BurstConfig | None = None,
        calibration: Calibration | None = None,
        fixed_gains: GainConfig | None = None,
        seed: int = 0,
    ):
        self.link = link or LinkConfig()
        self.codec = BurstCodec(self.link, burst or BurstConfig())
        self.backend = backend
        self.calibration = calibration
        self.fixed_gains = fixed_gains or GainConfig()
        self.rng = np.random.default_rng(seed)
        self.stats_log: list = []

    @property
    def k(self) -> int:
        return self.codec.k

    @property
    def last_stats(self) -> dict:
        return self.stats_log[-1] if self.stats_log else {}

    def payload_accounting(self) -> dict:
        """Honest per-message channel cost. `airComp/eval/metrics.py` counts the
        payload only; the sync and pilot overhead is real and is reported here
        so a bandwidth claim can never be made from payload alone.

        The overhead is a fixed per-BURST cost, so packing several negotiation
        turns into one burst would amortize it. We deliberately send one message
        per burst to keep the hardware path identical to the simulated one.
        """
        layout = self.codec.layout
        return {
            "data_symbols": layout.n_data,
            "overhead_symbols": layout.overhead_symbols,
            "total_symbols": layout.total_symbols,
            "burst_duration_s": layout.total_symbols / self.link.symbol_rate,
            "occupied_bandwidth_hz": self.link.symbol_rate * (1.0 + self.link.rolloff),
        }

    def _gains_for(self, snr_db: float) -> tuple:
        if self.calibration is None:
            return self.fixed_gains, None
        point = self.calibration.nearest(snr_db)
        return point.gains(), point

    def _transmit_one(self, z: np.ndarray, snr_db: float) -> np.ndarray:
        gains, point = self._gains_for(snr_db)
        self.backend.configure(gains)
        tx = self.codec.modulate(z)

        attempts = 0
        for attempts in range(1, self.link.max_retries + 1):
            decoded = self.codec.demodulate(self.backend.send_and_capture(tx, self.codec.capture_samples))
            if decoded is not None:
                self.stats_log.append(
                    {
                        "requested_snr_db": float(snr_db),
                        "calibrated_snr_db": float(point.measured_snr_db) if point else None,
                        "measured_snr_db": float(decoded.snr_db),
                        "pilot_snr_db": float(decoded.pilot_snr_db),
                        "snr_disagreement_db": float(decoded.snr_disagreement_db),
                        "channel_gain_abs": float(abs(decoded.h)),
                        "preamble_peak_ratio": float(decoded.peak_ratio),
                        "image_rejection_db": float(decoded.image_rejection_db),
                        "rx_peak_lsb": float(decoded.levels["peak_lsb"]),
                        "rx_warnings": list(decoded.levels["warnings"]),
                        "attempts": attempts,
                        "burst_lost": False,
                        "gains": vars(gains).copy(),
                    }
                )
                if decoded.levels["warnings"]:
                    warnings.warn("; ".join(decoded.levels["warnings"]), RuntimeWarning, stacklevel=2)
                return decoded.z_hat

        # Frame detection failed. The physically honest representation of "the
        # receiver got nothing" is noise with no signal in it -- not a retry
        # that pretends the transmission succeeded, and not a dropped episode.
        # The decoder still has to act on what arrived, exactly as it would on
        # the real link. This is counted, not hidden; loss_rate is reported.
        sigma = float(np.sqrt(noise_var_per_real(snr_db)))
        self.stats_log.append(
            {
                "requested_snr_db": float(snr_db),
                "calibrated_snr_db": float(point.measured_snr_db) if point else None,
                "measured_snr_db": None,
                "attempts": attempts,
                "burst_lost": True,
                "gains": vars(gains).copy(),
            }
        )
        warnings.warn(
            f"burst lost after {attempts} attempts at requested SNR {snr_db} dB; "
            f"delivering noise-only to the decoder",
            RuntimeWarning,
            stacklevel=2,
        )
        return self.rng.normal(0.0, sigma, size=self.k)

    def __call__(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        arr = z.detach().cpu().numpy().astype(float)
        squeeze = arr.ndim == 1
        rows = arr.reshape(1, -1) if squeeze else arr
        if rows.shape[-1] != self.k:
            raise ValueError(
                f"encoder produces k={rows.shape[-1]} but the burst carries k={self.k}; "
                f"set BurstConfig.n_data = k/2"
            )
        out = np.stack([self._transmit_one(row, snr_db) for row in rows])
        result = out[0] if squeeze else out
        return torch.as_tensor(result, dtype=z.dtype, device=z.device)

    def loss_rate(self) -> float:
        if not self.stats_log:
            return 0.0
        return float(np.mean([s["burst_lost"] for s in self.stats_log]))

    def close(self) -> None:
        self.backend.close()
