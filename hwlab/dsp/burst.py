"""BurstCodec -- ties mapping/framing/pulse/sync/equalize/measure together.

    z (k reals)  --modulate-->  full-scale IQ  --[ SDRBackend ]-->  IQ
                 <-demodulate--                                <---

`modulate` and `demodulate` are the only two entry points the channel layer
needs. Everything they rely on is unit-tested against the loopback backend, so
the DSP is verified before any hardware is powered on.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from hwlab.config import BurstConfig, LinkConfig
from hwlab.dsp import equalize, measure, pulse, sync
from hwlab.dsp.framing import BurstLayout
from hwlab.dsp.mapping import symbols_to_z, z_to_symbols


@dataclass
class BurstDecode:
    z_hat: np.ndarray
    h: complex
    #: Primary SNR estimate, from the guard-region noise floor.
    snr_db: float
    #: Independent cross-check, from the pilot residual. Noisier (only n_pilots
    #: samples) but measured exactly where the data sits. A large disagreement
    #: with `snr_db` means something is wrong -- typically clipping, a spur in
    #: band, or a one-sample timing error.
    pilot_snr_db: float
    noise_var_per_real: float
    peak_ratio: float
    image_rejection_db: float
    levels: dict

    @property
    def snr_disagreement_db(self) -> float:
        return abs(self.snr_db - self.pilot_snr_db)


class BurstCodec:
    def __init__(self, link: LinkConfig, burst: BurstConfig):
        self.link = link
        self.layout = BurstLayout(**asdict(burst))
        self.taps = pulse.rrc_taps(link.sps, link.span_symbols, link.rolloff)
        self.reference = sync.build_reference(self.layout.preamble(), link.sps, self.taps)
        self._pilots = self.layout.pilots()
        self.tx_scale = self._compute_tx_scale()

    # -- geometry -------------------------------------------------------

    @property
    def burst_samples(self) -> int:
        return self.layout.total_symbols * self.link.sps + len(self.taps) - 1

    @property
    def capture_samples(self) -> int:
        return int(math.ceil(self.link.capture_bursts * self.burst_samples))

    @property
    def k(self) -> int:
        return 2 * self.layout.n_data

    def _compute_tx_scale(self) -> float:
        """Fixed TX scaling, computed once from a representative burst.

        Deliberately NOT per-burst auto-scaling: normalizing each burst to the
        same peak would make transmit power depend on the payload, which is
        exactly the kind of hidden gain change that corrupts an SNR sweep.
        """
        rng = np.random.default_rng(1234)
        z = rng.normal(size=self.k)
        z *= math.sqrt(self.k) / np.linalg.norm(z)  # matches SemanticEncoder's ||z|| = sqrt(k)
        waveform = pulse.shape(self.layout.build(z_to_symbols(z)), self.link.sps, self.taps)
        peak = float(np.max(np.abs(waveform)))
        return self.link.dac_peak / peak if peak > 0 else 1.0

    # -- transmit -------------------------------------------------------

    def modulate(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != self.k:
            raise ValueError(f"expected k={self.k} reals, got {z.size}")
        symbols = z_to_symbols(z)
        waveform = pulse.shape(self.layout.build(symbols), self.link.sps, self.taps)
        return pulse.mix_up(waveform * self.tx_scale, self.link.fs, self.link.if_offset_hz)

    # -- receive --------------------------------------------------------

    def demodulate(self, rx_iq: np.ndarray) -> Optional[BurstDecode]:
        rx_iq = np.asarray(rx_iq, dtype=complex)
        levels = measure.level_report(rx_iq)
        if rx_iq.size < len(self.reference):
            return None

        # Remove the zero-IF DC spur. The matched filter would reject most of it
        # anyway (the signal sits at +if_offset_hz), but this is free.
        centered = rx_iq - np.mean(rx_iq)
        y = pulse.matched_filter(pulse.mix_down(centered, self.link.fs, self.link.if_offset_hz), self.taps)

        found = sync.find_burst(
            y,
            self.reference,
            self.layout.preamble_start,
            self.link.sps,
            self.taps,
            self.layout.total_symbols,
            self.link.min_peak_ratio,
        )
        if found is None:
            return None

        rx_preamble = sync.sample_symbols(
            y, found.sym0_index, self.link.sps, self.layout.preamble_start, self.layout.preamble_len
        )
        rx_pilots = sync.sample_symbols(y, found.sym0_index, self.link.sps, self.layout.pilot_start, self.layout.n_pilots)
        rx_data = sync.sample_symbols(y, found.sym0_index, self.link.sps, self.layout.data_start, self.layout.n_data)

        # Estimate h from the PREAMBLE, not the pilots. The preamble is already
        # being transmitted for sync and is 16x longer, which is ~12 dB more
        # averaging -- for free. With only 32 pilots the estimate of |h| swings
        # by 25 dB at -10 dB SNR, which both corrupts the reported SNR and
        # inflates the equalized noise. Leaving the pilots out of the fit also
        # keeps their residual an *unbiased* independent check; a fit residual
        # would flatter itself by absorbing part of the noise.
        h = equalize.estimate_gain(rx_preamble, self.layout.preamble())
        noise_var = measure.noise_var_per_real_from_guard(
            y, found.sym0_index, self.link.sps, self.layout.guard_slice()
        )
        z_hat = symbols_to_z(equalize.equalize(rx_data, h))

        pilot_residual = equalize.pilot_residual_noise_var_per_real(rx_pilots, self._pilots, h)
        return BurstDecode(
            z_hat=z_hat,
            h=h,
            snr_db=measure.measured_snr_db(h, noise_var),
            pilot_snr_db=measure.snr_db_from_equalized_residual(pilot_residual),
            noise_var_per_real=noise_var,
            peak_ratio=found.peak_ratio,
            image_rejection_db=equalize.iq_imbalance_metrics(rx_pilots, self._pilots)["image_rejection_db"],
            levels=levels,
        )
