# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Source coding for a negotiation message: Offer <-> a fixed 8-bit frame.

Why this exists
---------------
`TextAgent` transmits the LLM's entire completion -- prose preamble, the offer
JSON, and a free-text `"message"` rationale -- which measures 600-1350 bits per
message. The communicative act inside it is worth **6.1 bits**: both agents
already know the pool, so an offer is an index into the feasible count-vectors
for that pool (16-36 of them under the default config) plus one of three
actions.

Comparing a 16-real-symbol semantic latent against ~1000 bits of prose conflates
source coding with joint source-channel coding, and the source-coding term
dominates. This codec supplies the missing control: the same LLM turn, the same
BPSK/AWGN channel, but only the communicative act on the wire.

Fairness
--------
The codec conditions on `pool`, which is shared, public knowledge. That is
exactly the information `pool_to_mask` (airComp/jscc/modules.py) gives the
semantic decoder, so neither side gets a private advantage from it.

Frame
-----
    bit  0-1 : action   (0=propose, 1=accept, 2=reject; 3 is invalid)
    bit  2-7 : index into the feasible count-vectors, MSB first

Fixed width regardless of pool, so the channel-use budget does not vary with the
draw. 6 index bits cover 64 count-vectors; the default config peaks at 36.
"""
from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np

from airComp.env.negotiation import Offer, Pool

ACTIONS = ("propose", "accept", "reject")

ACTION_BITS = 2
INDEX_BITS = 6
#: Total frame width. Also the channel-use count for the uncoded compact baseline.
OFFER_BITS = ACTION_BITS + INDEX_BITS

MAX_FEASIBLE = 1 << INDEX_BITS


def feasible_counts(pool: Pool) -> list:
    """Every count-vector a proposer may keep, in a canonical order.

    Sorted item names rather than config order: the sender and receiver must
    agree, and sorting makes that agreement independent of how the Pool was
    built.
    """
    types = sorted(pool.counts)
    ranges = [range(pool.counts[t] + 1) for t in types]
    return [dict(zip(types, combo)) for combo in product(*ranges)]


def _int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def offer_to_bits(offer: Offer, pool: Pool) -> np.ndarray:
    """Encode an offer as `OFFER_BITS` bits. Raises if the offer is not sendable.

    The rationale (`Offer.message`) is deliberately dropped: the semantic
    pipeline never transmits it either, so putting it on the wire here would
    reintroduce the asymmetry this codec exists to remove.
    """
    if offer.action not in ACTIONS:
        raise ValueError(f"unknown action {offer.action!r}")

    index = 0
    if offer.action == "propose":
        options = feasible_counts(pool)
        if len(options) > MAX_FEASIBLE:
            raise ValueError(
                f"pool has {len(options)} feasible offers, more than {INDEX_BITS} index bits can address"
            )
        try:
            index = options.index({t: int(offer.counts[t]) for t in sorted(pool.counts)})
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"counts {offer.counts!r} are not feasible for pool {pool.counts!r}")

    return np.concatenate([
        _int_to_bits(ACTIONS.index(offer.action), ACTION_BITS),
        _int_to_bits(index, INDEX_BITS),
    ])


def bits_to_offer(bits: np.ndarray, pool: Pool) -> Optional[Offer]:
    """Decode a received frame, or None if it does not name a valid offer.

    None is the digital counterpart of a failed JSON parse, and the caller
    treats it the same way -- as an implicit REJECT. Unlike a corrupted latent,
    a corrupted index has no locality: one flipped bit selects an unrelated
    count-vector rather than a neighbouring one. That contrast is the point of
    the comparison, so it must not be smoothed over here.
    """
    if bits is None or len(bits) < OFFER_BITS:
        return None

    action_idx = _bits_to_int(bits[:ACTION_BITS])
    if action_idx >= len(ACTIONS):
        return None
    action = ACTIONS[action_idx]
    if action != "propose":
        return Offer(action=action, counts=None, message=None)

    options = feasible_counts(pool)
    index = _bits_to_int(bits[ACTION_BITS:OFFER_BITS])
    if index >= len(options):
        return None
    return Offer(action="propose", counts=options[index], message=None)
