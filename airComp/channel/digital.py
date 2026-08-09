"""Simulated conventional digital channel: UTF-8 text -> bits -> BPSK -> AWGN -> demod -> text.

Two modes:
  "raw": no FEC, no retransmission. A single flipped bit inside a JSON
         structural character can break parsing entirely -- this is what
         demonstrates the catastrophic failure cliff of naive digital text
         communication under noise.
  "arq": CRC-8 appended; a failed checksum means the message is dropped
         (treated as lost, not silently corrupted) -- the more realistic
         "modern digital comms" baseline (error-detected-and-discarded).
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
        assert mode in ("raw", "arq")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def transmit(self, text: str, snr_db: float):
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
