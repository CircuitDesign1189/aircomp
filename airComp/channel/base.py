# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Abstract channel interface.

A future SDR-backed implementation (real RF via GNU Radio / USRP / HackRF /
LimeSDR) can implement this same interface without touching agent or task
code. Nothing in this prototype implements real hardware transmission yet --
see the "Explicitly out of scope" section in CLAUDE.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Channel(ABC):
    @abstractmethod
    def transmit(self, payload: Any, snr_db: float):
        """Send payload through the channel.

        Returns (received_payload_or_None, stats) where `received_payload`
        is None if the message was lost/undecodable (e.g. failed CRC).
        """
        raise NotImplementedError
