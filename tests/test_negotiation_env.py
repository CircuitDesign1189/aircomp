import numpy as np
import pytest

from airComp.config import NegotiationConfig
from airComp.env.negotiation import Offer, TurnRecord, generate_pool, generate_values, run_episode


def test_generate_pool_bounds():
    cfg = NegotiationConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        pool = generate_pool(rng, cfg)
        assert pool.total in cfg.total_items_choices
        for t in cfg.item_types:
            assert cfg.min_per_type <= pool.counts[t] <= cfg.max_per_type


def test_generate_values_full_pool_worth_100_points():
    cfg = NegotiationConfig()
    rng = np.random.default_rng(1)
    for _ in range(50):
        pool = generate_pool(rng, cfg)
        values = generate_values(rng, pool, cfg)
        assert values.utility(pool.counts) == pytest.approx(cfg.pool_value_points, rel=1e-6)


class ScriptedAgent:
    """Plays a fixed script of Offers, one per call to take_turn."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def take_turn(self, pool, own_values, round_index, history, standing_offer):
        offer = self.script[self.calls]
        self.calls += 1
        return TurnRecord(raw_offer=offer, received_offer=offer, parse_failed=False, channel_stats={})


def test_immediate_reject_ends_no_deal():
    cfg = NegotiationConfig(max_messages=10)
    agent_a = ScriptedAgent([Offer(action="reject")])
    agent_b = ScriptedAgent([Offer(action="reject")])
    record = run_episode(agent_a, agent_b, seed=42, cfg=cfg)
    assert record.outcome == "no_deal"
    assert record.utility_a == 0.0
    assert record.utility_b == 0.0


class ProposeThenAcceptAgent:
    """Proposes fixed counts if there's no standing offer yet, otherwise accepts.

    Using this (rather than an index-based script) for both seats makes the
    test's expected outcome independent of which agent the environment
    happens to pick as the first mover.
    """

    def __init__(self, propose_counts):
        self.propose_counts = propose_counts

    def take_turn(self, pool, own_values, round_index, history, standing_offer):
        if standing_offer is not None:
            offer = Offer(action="accept")
        else:
            offer = Offer(action="propose", counts=dict(self.propose_counts))
        return TurnRecord(raw_offer=offer, received_offer=offer, parse_failed=False, channel_stats={})


def test_propose_then_accept_reaches_agreement():
    cfg = NegotiationConfig(max_messages=10)
    keep_nothing = {"book": 0, "hat": 0, "ball": 0}

    agent_a = ProposeThenAcceptAgent(keep_nothing)
    agent_b = ProposeThenAcceptAgent(keep_nothing)

    record = run_episode(agent_a, agent_b, seed=7, cfg=cfg)
    assert record.outcome == "agreement"
    proposer_counts = record.final_counts_a if record.first_mover == "A" else record.final_counts_b
    other_counts = record.final_counts_b if record.first_mover == "A" else record.final_counts_a
    assert proposer_counts == keep_nothing
    assert other_counts == record.pool.counts


def test_no_agreement_after_max_messages():
    cfg = NegotiationConfig(max_messages=4)
    keep_nothing = {"book": 0, "hat": 0, "ball": 0}
    propose = Offer(action="propose", counts=keep_nothing)
    agent_a = ScriptedAgent([propose, propose])
    agent_b = ScriptedAgent([propose, propose])
    record = run_episode(agent_a, agent_b, seed=3, cfg=cfg)
    assert record.outcome == "no_deal"
    assert len(record.turns) == cfg.max_messages
    assert record.utility_a == 0.0
    assert record.utility_b == 0.0


def test_accept_without_standing_offer_is_no_deal():
    cfg = NegotiationConfig(max_messages=10)
    agent_a = ScriptedAgent([Offer(action="accept")])
    agent_b = ScriptedAgent([Offer(action="accept")])
    record = run_episode(agent_a, agent_b, seed=5, cfg=cfg)
    assert record.outcome == "no_deal"


def test_parse_failure_is_implicit_reject():
    cfg = NegotiationConfig(max_messages=10)
    agent_b = ScriptedAgent([Offer(action="reject")])

    class ParseFailureAgent:
        def take_turn(self, pool, own_values, round_index, history, standing_offer):
            return TurnRecord(raw_offer=None, received_offer=None, parse_failed=True, channel_stats={})

    record = run_episode(ParseFailureAgent(), agent_b, seed=9, cfg=cfg)
    assert record.outcome == "no_deal"
    assert len(record.turns) == 1
