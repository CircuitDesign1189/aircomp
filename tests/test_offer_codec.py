"""The compact baseline is only a fair control if its codec is exact.

If encode/decode loses information, the compact pipeline looks worse than
digital transmission really is, and the semantic pipeline gets credit it has not
earned. So the round-trip is pinned exhaustively rather than sampled.
"""
from __future__ import annotations

import numpy as np
import pytest

from airComp.baseline.offer_codec import (
    MAX_FEASIBLE,
    OFFER_BITS,
    bits_to_offer,
    feasible_counts,
    offer_to_bits,
)
from airComp.config import NegotiationConfig
from airComp.env.negotiation import Offer, Pool, generate_pool


def _pools():
    cfg = NegotiationConfig()
    rng = np.random.default_rng(0)
    seen = {}
    for _ in range(2000):
        p = generate_pool(rng, cfg)
        seen[tuple(sorted(p.counts.items()))] = p
    return list(seen.values())


def test_every_feasible_offer_survives_a_round_trip():
    """Exhaustive over every pool shape the default config can draw."""
    for pool in _pools():
        for counts in feasible_counts(pool):
            offer = Offer(action="propose", counts=counts)
            decoded = bits_to_offer(offer_to_bits(offer, pool), pool)
            assert decoded is not None
            assert decoded.action == "propose"
            assert decoded.counts == counts


def test_accept_and_reject_round_trip():
    pool = Pool(counts={"book": 2, "hat": 2, "ball": 2})
    for action in ("accept", "reject"):
        decoded = bits_to_offer(offer_to_bits(Offer(action=action), pool), pool)
        assert decoded.action == action
        assert decoded.counts is None


def test_the_frame_is_a_fixed_width_for_every_pool():
    """A pool-dependent width would make the channel-use budget vary with the
    draw, which would quietly break the matched comparison against k=16."""
    for pool in _pools():
        assert len(offer_to_bits(Offer(action="propose", counts=feasible_counts(pool)[0]), pool)) == OFFER_BITS


def test_the_index_field_is_wide_enough_for_the_configured_pools():
    assert max(len(feasible_counts(p)) for p in _pools()) <= MAX_FEASIBLE


def test_the_rationale_is_not_transmitted():
    """The semantic pipeline never sends free text; the compact one must not either,
    or the payload asymmetry this codec exists to remove comes straight back."""
    pool = Pool(counts={"book": 1, "hat": 1, "ball": 1})
    offer = Offer(action="propose", counts={"ball": 1, "book": 0, "hat": 1}, message="I value hats")

    decoded = bits_to_offer(offer_to_bits(offer, pool), pool)

    assert decoded.message is None
    assert decoded.counts == {"ball": 1, "book": 0, "hat": 1}


def test_an_out_of_range_index_decodes_to_none():
    """None is the digital equivalent of a failed parse -- an implicit REJECT."""
    pool = Pool(counts={"book": 1, "hat": 1, "ball": 1})  # 8 feasible, indices 8..63 invalid
    bits = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

    assert bits_to_offer(bits, pool) is None


def test_an_invalid_action_code_decodes_to_none():
    pool = Pool(counts={"book": 1, "hat": 1, "ball": 1})
    bits = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)  # action 3 does not exist

    assert bits_to_offer(bits, pool) is None


def test_infeasible_counts_are_rejected_at_encode_time():
    pool = Pool(counts={"book": 1, "hat": 1, "ball": 1})

    with pytest.raises(ValueError):
        offer_to_bits(Offer(action="propose", counts={"book": 5, "hat": 0, "ball": 0}), pool)


def test_how_a_single_bit_flip_displaces_the_decoded_offer():
    """Characterises the mechanism the whole comparison turns on -- and corrects
    an assumption that turned out to be wrong.

    A lexicographic index over a rectangular count-grid is a mixed-radix
    positional code, so it is *not* locality-free: flipping a digit's low bit
    moves one item type by one. Measured over every pool the default config can
    draw, single index-bit flips land as

        24.5%  outside the feasible set -> implicit REJECT (total loss)
        26.8%  at L1 distance 1          (mild)
        48.7%  at L1 distance >= 2

    So the compact baseline is a genuinely strong control, not a straw man: it
    degrades partly gracefully. What it cannot do is avoid the quarter of flips
    that fall off the codebook entirely, which is the failure mode an analog
    latent has no equivalent of.
    """
    invalid = near = total = 0
    for pool in _pools():
        for counts in feasible_counts(pool):
            clean = offer_to_bits(Offer(action="propose", counts=counts), pool)
            for bit in range(2, OFFER_BITS):
                flipped = clean.copy()
                flipped[bit] ^= 1
                got = bits_to_offer(flipped, pool)
                total += 1
                if got is None:
                    invalid += 1
                elif sum(abs(got.counts[t] - counts[t]) for t in counts) <= 1:
                    near += 1

    assert 0.15 < invalid / total < 0.35, "a sizeable share must fall off the codebook"
    assert near / total < 0.5, "but the index is not so local that noise is harmless"
