# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Convert an EpisodeRecord into a JSON-serializable dict for episode logs."""
from __future__ import annotations

from airComp.env.negotiation import EpisodeRecord


def episode_record_to_dict(record: EpisodeRecord) -> dict:
    return {
        "seed": record.seed,
        "pool": record.pool.counts,
        "values_a": record.values_a.per_unit,
        "values_b": record.values_b.per_unit,
        "first_mover": record.first_mover,
        "outcome": record.outcome,
        "rounds_used": len(record.turns),
        "final_counts_a": record.final_counts_a,
        "final_counts_b": record.final_counts_b,
        "utility_a": record.utility_a,
        "utility_b": record.utility_b,
        "turns": [
            {
                "turn_index": t.turn_index,
                "agent": t.agent,
                "received_action": t.received_offer.action if t.received_offer else None,
                "received_counts": t.received_offer.counts if t.received_offer else None,
                "parse_failed": t.parse_failed,
                "channel_stats": t.channel_stats,
            }
            for t in record.turns
        ],
    }
