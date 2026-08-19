"""Simulated conventional digital channel: UTF-8 text -> bits -> BPSK -> AWGN -> demod -> text.

Two modes:
  "raw": no FEC, no retransmission. A single flipped bit inside a JSON
         structural character can break parsing entirely -- this is what
         demonstrates the catastrophic failure cliff of naive digital text
         communication under noise.
  "arq": CRC-8 appended; a failed checksum means the message is dropped
         (treated as lost, not silently corrupted) -- error-detected-and-discarded.
  "fec": Hamming(7,4) forward error correction over a short bit frame. Reached
         through `transmit_bits`, not `transmit`: it exists for the compact
         baseline, whose payload is the 8-bit offer frame rather than text.

Note that neither "raw" nor "arq" corrects errors -- CRC only detects them. A
real digital link always carries FEC, so "fec" over the compact frame, not
"arq" over prose, is the honest "modern digital comms" baseline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from airComp.channel.base import Channel


def _crc8(data: bytes) -> int:
    crc = 0x00
    poly = 0x07
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


#: Channel uses per message for the FEC-coded compact frame. Two Hamming(7,4)
#: blocks carry the 8-bit offer frame in 14 bits; two zero bits pad the burst to
#: 16 so that the compact baseline occupies **exactly the same number of real
#: channel uses as the semantic pipeline's k=16 latent**, at the same SNR per
#: real dimension. That equality is what makes the two directly comparable.
FEC_FRAME_BITS = 16

_H74_PARITY = ((0, 1, 3), (0, 2, 3), (1, 2, 3))  # data indices feeding p1, p2, p3
_H74_LAYOUT = ("p0", "p1", 0, "p2", 1, 2, 3)  # codeword positions 1..7


def hamming74_encode(nibble: np.ndarray) -> np.ndarray:
    """4 data bits -> a 7-bit codeword that corrects any single-bit error."""
    parity = [int(np.bitwise_xor.reduce(nibble[list(idx)])) for idx in _H74_PARITY]
    return np.array(
        [parity[int(s[1])] if isinstance(s, str) else int(nibble[s]) for s in _H74_LAYOUT],
        dtype=np.uint8,
    )


def hamming74_decode(codeword: np.ndarray) -> np.ndarray:
    """7 bits -> the 4 data bits, correcting a single-bit error if present.

    Two or more errors are not detected; the syndrome points at an innocent bit
    and decoding silently returns the wrong nibble. That is the correct
    behaviour to model -- a real short block code degrades this way too.
    """
    word = codeword.astype(np.uint8).copy()
    data = np.array([word[i] for i in (2, 4, 5, 6)], dtype=np.uint8)
    parity = [word[0], word[1], word[3]]
    syndrome = 0
    for bit, idx in enumerate(_H74_PARITY):
        s = int(parity[bit]) ^ int(np.bitwise_xor.reduce(data[list(idx)]))
        syndrome |= s << bit
    if syndrome:
        word[syndrome - 1] ^= 1
    return np.array([word[i] for i in (2, 4, 5, 6)], dtype=np.uint8)


def text_to_bits(text: str) -> np.ndarray:
    data = text.encode("utf-8")
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_text(bits: np.ndarray) -> str:
    n = (len(bits) // 8) * 8
    if n == 0:
        return ""
    byte_arr = np.packbits(bits[:n])
    return byte_arr.tobytes().decode("utf-8", errors="replace")


def bpsk_awgn(bits: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(bits) == 0:
        return bits.copy()
    symbols = np.where(bits == 1, 1.0, -1.0)
    sigma = float(10 ** (-snr_db / 10.0)) ** 0.5
    noise = rng.normal(0.0, sigma, size=symbols.shape)
    received = symbols + noise
    return (received > 0).astype(np.uint8)


class DigitalChannel(Channel):
    def __init__(self, mode: str = "raw", seed: Optional[int] = None):
        assert mode in ("raw", "arq", "fec")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def transmit_bits(self, bits: np.ndarray, snr_db: float):
        """Send a short bit frame -- the compact baseline's path.

        mode="raw": the frame goes on the wire uncoded.
        mode="fec": Hamming(7,4) per nibble, zero-padded to FEC_FRAME_BITS.

        Returns (received_bits_or_None, stats). Unlike the text path this never
        returns None -- a short frame is always demodulated to *something*, and
        whether it names a valid offer is the codec's judgement, not the
        channel's.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if self.mode == "arq":
            raise ValueError("arq is a text-frame mode; use 'raw' or 'fec' for bit frames")

        if self.mode == "fec":
            if len(bits) % 4:
                raise ValueError(f"fec needs a whole number of nibbles, got {len(bits)} bits")
            coded = np.concatenate([hamming74_encode(bits[i:i + 4]) for i in range(0, len(bits), 4)])
            sent = np.zeros(FEC_FRAME_BITS, dtype=np.uint8)
            if len(coded) > FEC_FRAME_BITS:
                raise ValueError(f"{len(coded)} coded bits exceed the {FEC_FRAME_BITS}-use budget")
            sent[: len(coded)] = coded
        else:
            sent = bits

        received = bpsk_awgn(sent, snr_db, self.rng)
        channel_errors = int(np.sum(received != sent))
        stats = {
            "mode": self.mode,
            "snr_db": snr_db,
            "bit_errors": channel_errors,
            "ber": channel_errors / len(sent) if len(sent) else 0.0,
            "n_bits": int(len(sent)),  # channel uses, so the accounting stays honest
        }

        if self.mode != "fec":
            return received, stats

        decoded = np.concatenate(
            [hamming74_decode(received[i:i + 7]) for i in range(0, len(coded), 7)]
        )
        stats["residual_bit_errors"] = int(np.sum(decoded != bits))
        stats["fec_corrected"] = channel_errors - stats["residual_bit_errors"]
        return decoded, stats

    def transmit(self, text: str, snr_db: float):
        if self.mode == "fec":
            raise ValueError("fec is a bit-frame mode; use transmit_bits")
        payload = text.encode("utf-8")

        if self.mode == "arq":
            checksum = _crc8(payload)
            payload_with_crc = payload + bytes([checksum])
            bits = np.unpackbits(np.frombuffer(payload_with_crc, dtype=np.uint8)) if payload_with_crc else np.zeros(0, dtype=np.uint8)
        else:
            bits = text_to_bits(text)

        received_bits = bpsk_awgn(bits, snr_db, self.rng)
        bit_errors = int(np.sum(received_bits != bits))
        ber = bit_errors / len(bits) if len(bits) > 0 else 0.0
        stats = {"mode": self.mode, "snr_db": snr_db, "bit_errors": bit_errors, "ber": ber, "n_bits": int(len(bits))}

        if self.mode == "arq":
            n = (len(received_bits) // 8) * 8
            received_bytes = np.packbits(received_bits[:n]).tobytes() if n > 0 else b""
            if len(received_bytes) < 1:
                stats["crc_ok"] = False
                return None, stats
            recv_payload, recv_crc = received_bytes[:-1], received_bytes[-1]
            crc_ok = _crc8(recv_payload) == recv_crc
            stats["crc_ok"] = crc_ok
            if not crc_ok:
                return None, stats
            return recv_payload.decode("utf-8", errors="replace"), stats

        return bits_to_text(received_bits), stats
