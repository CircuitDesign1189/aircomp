# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""SDRDigitalChannel -- the fair digital baseline over real RF.

Same call signature as `airComp.channel.digital.DigitalChannel.transmit_bits`,
so `CompactAgent` plugs into the radio by having its `channel` attribute
swapped, exactly as `HardwareSemanticAgent` does for the analog path.

Why this fits the existing burst with no changes
------------------------------------------------
`compact_fec` sends 8 offer bits under Hamming(7,4) padded to 16 -- and 16 BPSK
bits are 16 real dimensions, which pair into the same **8 complex symbols** the
k=16 semantic latent uses. `BurstConfig.n_data` stays at 8 and
`hwlab/configs/sdr_link.yaml` is untouched.

The power convention lines up for free. `hwlab/dsp/mapping.py` fixes
SIGNAL_POWER_PER_REAL = 1.0 because a power-normalized z has unit power per real
component; BPSK +-1 has *exactly* unit power per real component. Nothing is
rescaled -- which matters, because the 1/sqrt(2) rescaling that module warns
about is precisely how a 3 dB error would creep into the digital path and make
the hardware-vs-simulation comparison meaningless.

`BurstCodec.tx_scale` is computed once from a representative Gaussian burst and
is deliberately not per-burst (hwlab/dsp/burst.py), so switching the payload to
a constant-modulus one does not change transmit power either.

A lost burst delivers noise, the hard decision turns that into random bits, and
`bits_to_offer` rejects the resulting index -- an implicit REJECT. That is the
same thing the simulated compact pipeline does with a destroyed frame, and it is
the physically honest outcome: the receiver got nothing.
"""
from __future__ import annotations

import numpy as np

from airComp.channel.digital import FEC_FRAME_BITS, hamming74_decode, hamming74_encode
from hwlab.channel.sdr_analog import SDRAnalogChannel


class SDRDigitalChannel:
    def __init__(self, analog: SDRAnalogChannel, mode: str = "fec"):
        if mode != "fec":
            raise ValueError(
                f"only 'fec' is carried over the radio, got {mode!r}. The uncoded compact "
                f"frame is 8 bits = 4 complex symbols, which needs a different BurstConfig "
                f"and so cannot share this burst with the k=16 latent."
            )
        if analog.k != FEC_FRAME_BITS:
            raise ValueError(
                f"the burst carries k={analog.k} reals but the coded frame is "
                f"{FEC_FRAME_BITS} bits; set burst.n_data = {FEC_FRAME_BITS // 2}"
            )
        self.analog = analog
        self.mode = mode

    @property
    def last_stats(self) -> dict:
        return self.analog.last_stats

    def payload_accounting(self) -> dict:
        return self.analog.payload_accounting()

    def loss_rate(self) -> float:
        return self.analog.loss_rate()

    def close(self) -> None:
        self.analog.close()

    def transmit_bits(self, bits: np.ndarray, snr_db: float):
        """8 information bits -> 16 channel uses over the radio -> 8 bits."""
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if len(bits) % 4:
            raise ValueError(f"fec needs a whole number of nibbles, got {len(bits)} bits")

        coded = np.concatenate([hamming74_encode(bits[i:i + 4]) for i in range(0, len(bits), 4)])
        if len(coded) > FEC_FRAME_BITS:
            raise ValueError(f"{len(coded)} coded bits exceed the {FEC_FRAME_BITS}-use budget")
        sent = np.zeros(FEC_FRAME_BITS, dtype=np.uint8)
        sent[: len(coded)] = coded

        # BPSK: 0 -> -1, 1 -> +1. Unit power per real, no scaling factor.
        received_reals = self.analog.transmit_reals(sent.astype(float) * 2.0 - 1.0, snr_db)
        received = (np.asarray(received_reals) > 0).astype(np.uint8)

        decoded = np.concatenate(
            [hamming74_decode(received[i:i + 7]) for i in range(0, len(coded), 7)]
        )

        stats = dict(self.analog.last_stats)
        channel_errors = int(np.sum(received != sent))
        stats.update(
            {
                "mode": self.mode,
                "snr_db": snr_db,
                "n_bits": FEC_FRAME_BITS,  # channel uses, matching the latent's k=16
                "bit_errors": channel_errors,
                "residual_bit_errors": int(np.sum(decoded != bits)),
            }
        )
        stats["fec_corrected"] = channel_errors - stats["residual_bit_errors"]
        return decoded, stats
