# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Integration test requiring the actual local LLM to be downloaded.
Run with `pytest -m slow -q`.
"""
import pytest

from airComp.agents.baseline_agent import TextAgent
from airComp.agents.llm_backend import LocalLLM
from airComp.channel.digital import DigitalChannel
from airComp.config import NegotiationConfig
from airComp.env.negotiation import run_episode


@pytest.mark.slow
def test_full_baseline_episode_runs_end_to_end():
    llm = LocalLLM(device="cpu")  # force CPU so this doesn't require a GPU to run
    cfg = NegotiationConfig(max_messages=6)
    agent_a = TextAgent(llm, DigitalChannel(mode="raw", seed=0), snr_db=20.0, max_messages=cfg.max_messages)
    agent_b = TextAgent(llm, DigitalChannel(mode="raw", seed=1), snr_db=20.0, max_messages=cfg.max_messages)
    record = run_episode(agent_a, agent_b, seed=123, cfg=cfg)
    assert record.outcome in ("agreement", "no_deal")
    assert len(record.turns) <= cfg.max_messages
