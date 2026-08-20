# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""HardwareSemanticAgent -- SemanticAgent with its channel swapped for real RF.

`airComp/` is intentionally left untouched. `SemanticAgent.__init__` hardcodes
`self.channel = AnalogAWGNChannel()` (airComp/agents/semantic_agent.py:45) but
`take_turn` only ever calls `self.channel(z, self.snr_db)` under
`@torch.no_grad()`, so replacing the attribute after construction is sufficient
and needs no gradient support.

If the hardware path is ever promoted out of `hwlab`, the tidier form is a
`channel=` argument on SemanticAgent with AnalogAWGNChannel as the default.
Until then this keeps the published simulation results untouchable.
"""
from __future__ import annotations

from airComp.agents.compact_agent import CompactAgent
from airComp.agents.semantic_agent import SemanticAgent
from airComp.env.negotiation import TurnRecord


class HardwareSemanticAgent(SemanticAgent):
    def __init__(self, *args, channel, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_k = getattr(self.encoder, "k", None)
        channel_k = getattr(channel, "k", None)
        if encoder_k is not None and channel_k is not None and encoder_k != channel_k:
            raise ValueError(
                f"encoder emits k={encoder_k} reals but the burst carries k={channel_k}; "
                f"set BurstConfig.n_data = {encoder_k // 2}"
            )
        self.channel = channel

    def take_turn(self, *args, **kwargs) -> TurnRecord:
        turn = super().take_turn(*args, **kwargs)
        # Attach what the radio actually did, so the sweep can label its x-axis
        # with measured SNR instead of the requested value.
        turn.channel_stats.update(getattr(self.channel, "last_stats", {}))
        return turn


class HardwareCompactAgent(CompactAgent):
    """CompactAgent with its DigitalChannel swapped for real RF.

    The fair digital baseline over the radio. `CompactAgent.take_turn` only ever
    calls `self.channel.transmit_bits(bits, snr_db)`, so an `SDRDigitalChannel`
    -- which duck-types that method -- is a drop-in, and `airComp/` stays
    untouched exactly as it does for the semantic path above.

    Both agents in an episode share one channel object, and therefore one radio:
    the bench has two HackRFs but they are one TX/RX pair, not one pair per
    agent. The stats are logged per burst, so the sweep can still attribute
    them.
    """

    def __init__(self, *args, channel, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = channel

    def take_turn(self, *args, **kwargs) -> TurnRecord:
        turn = super().take_turn(*args, **kwargs)
        # Only if the channel was actually used -- an unparseable generation
        # skips it, and attaching a previous burst's stats would misreport it.
        if not turn.channel_stats.get("skipped_channel"):
            turn.channel_stats.update(getattr(self.channel, "last_stats", {}))
        return turn
