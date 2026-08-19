"""Deal-or-No-Deal-style bilateral bargaining environment.

Two agents negotiate over a shared pool of items (book/hat/ball). Each agent has
private, independently-randomized per-item values (normalized so taking the
entire pool is worth 100 points). Agents alternate PROPOSE/ACCEPT/REJECT
messages; running out of rounds or an explicit REJECT ends the episode with
zero utility for both agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import numpy as np

from airComp.config import NegotiationConfig

Action = Literal["propose", "accept", "reject"]
AgentId = Literal["A", "B"]


@dataclass(frozen=True)
class Pool:
    counts: dict

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class Values:
    per_unit: dict

    def utility(self, counts_received: dict) -> float:
        return sum(counts_received.get(t, 0) * v for t, v in self.per_unit.items())


def utility(counts_received: dict, values: Values) -> float:
    return values.utility(counts_received)


@dataclass
class Offer:
    action: Action
    counts: Optional[dict] = None  # amount the PROPOSER wants to keep; None for accept/reject
    message: Optional[str] = None


def validate_offer_counts(pool: Pool, counts: Optional[dict]) -> bool:
    if counts is None:
        return False
    for t, pool_count in pool.counts.items():
        c = counts.get(t)
        if c is None or not isinstance(c, int) or not (0 <= c <= pool_count):
            return False
    return True


@dataclass
class TurnRecord:
    turn_index: int = -1
    agent: Optional[AgentId] = None
    raw_offer: Optional[Offer] = None
    received_offer: Optional[Offer] = None
    parse_failed: bool = False
    channel_stats: dict = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    seed: int
    pool: Pool
    values_a: Values
    values_b: Values
    first_mover: AgentId
    turns: list = field(default_factory=list)
    outcome: Literal["agreement", "no_deal"] = "no_deal"
    final_counts_a: Optional[dict] = None
    final_counts_b: Optional[dict] = None
    utility_a: float = 0.0
    utility_b: float = 0.0


class Agent(Protocol):
    """Interface implemented by TextAgent and SemanticAgent."""

    def take_turn(
        self,
        pool: Pool,
        own_values: Values,
        round_index: int,
        history: list,
        standing_offer: Optional[Offer],
    ) -> TurnRecord: ...


def generate_pool(rng: np.random.Generator, cfg: NegotiationConfig = NegotiationConfig()) -> Pool:
    item_types = list(cfg.item_types)
    n = len(item_types)
    total = int(rng.choice(cfg.total_items_choices))
    counts = {}
    remaining = total
    for i, t in enumerate(item_types):
        remaining_slots = n - i - 1
        lo = max(cfg.min_per_type, remaining - remaining_slots * cfg.max_per_type)
        hi = min(cfg.max_per_type, remaining - remaining_slots * cfg.min_per_type)
        if lo > hi:
            # Chosen total is infeasible for the min/max-per-type bounds; retry with a fresh total.
            return generate_pool(rng, cfg)
        c = int(rng.integers(lo, hi + 1))
        counts[t] = c
        remaining -= c
    return Pool(counts=counts)


def generate_values(rng: np.random.Generator, pool: Pool, cfg: NegotiationConfig = NegotiationConfig()) -> Values:
    while True:
        weights = {t: rng.uniform(0, cfg.max_value_per_type) for t in cfg.item_types}
        denom = sum(w * pool.counts[t] for t, w in weights.items())
        if denom > 0:
            break
    per_unit = {t: w * cfg.pool_value_points / denom for t, w in weights.items()}
    return Values(per_unit=per_unit)


def _finalize_agreement(record: EpisodeRecord, pool: Pool, proposer: AgentId, proposer_keeps: dict) -> None:
    other = "B" if proposer == "A" else "A"
    other_gets = {t: pool.counts[t] - proposer_keeps.get(t, 0) for t in pool.counts}
    counts = {proposer: dict(proposer_keeps), other: other_gets}
    record.final_counts_a = counts["A"]
    record.final_counts_b = counts["B"]
    record.utility_a = record.values_a.utility(record.final_counts_a)
    record.utility_b = record.values_b.utility(record.final_counts_b)


def _finalize_no_deal(record: EpisodeRecord, pool: Pool) -> None:
    zero = {t: 0 for t in pool.counts}
    record.final_counts_a = dict(zero)
    record.final_counts_b = dict(zero)
    record.utility_a = 0.0
    record.utility_b = 0.0


def run_episode(
    agent_a: Agent,
    agent_b: Agent,
    seed: int,
    cfg: NegotiationConfig = NegotiationConfig(),
) -> EpisodeRecord:
    rng = np.random.default_rng(seed)
    pool = generate_pool(rng, cfg)
    values_a = generate_values(rng, pool, cfg)
    values_b = generate_values(rng, pool, cfg)
    first_mover: AgentId = "A" if rng.integers(0, 2) == 0 else "B"
    order = ["A", "B"] if first_mover == "A" else ["B", "A"]

    record = EpisodeRecord(seed=seed, pool=pool, values_a=values_a, values_b=values_b, first_mover=first_mover)
    agents = {"A": agent_a, "B": agent_b}
    values = {"A": values_a, "B": values_b}

    standing_offer: Optional[Offer] = None
    standing_offer_proposer: Optional[AgentId] = None

    for turn_index in range(cfg.max_messages):
        active: AgentId = order[turn_index % 2]
        turn = agents[active].take_turn(
            pool=pool,
            own_values=values[active],
            round_index=turn_index,
            history=list(record.turns),
            standing_offer=standing_offer,
        )
        turn.turn_index = turn_index
        turn.agent = active
        record.turns.append(turn)

        offer = turn.received_offer
        if offer is None:
            if not cfg.lost_message_ends_episode:
                # The turn is spent but the negotiation continues; the standing
                # offer is untouched and the next speaker sees the gap in
                # `history_prompt`. Without this, a digital pipeline gets exactly
                # one chance per episode while the semantic pipeline -- which
                # cannot produce an undecodable message -- gets max_messages.
                continue
            record.outcome = "no_deal"
            break
        if offer.action == "reject":
            record.outcome = "no_deal"
            break
        if offer.action == "accept":
            if standing_offer is None or standing_offer_proposer is None:
                record.outcome = "no_deal"
                break
            record.outcome = "agreement"
            _finalize_agreement(record, pool, standing_offer_proposer, standing_offer.counts)
            break
        if offer.action == "propose" and validate_offer_counts(pool, offer.counts):
            standing_offer = offer
            standing_offer_proposer = active
            continue
        # propose with invalid/out-of-range counts is treated as a failed turn (implicit reject)
        record.outcome = "no_deal"
        break
    else:
        record.outcome = "no_deal"

    if record.outcome == "no_deal" and record.final_counts_a is None:
        _finalize_no_deal(record, pool)

    return record
