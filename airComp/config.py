"""Config dataclasses for the AirComp negotiation prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ITEM_TYPES = ("book", "hat", "ball")


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    #: "torch" runs everything on CPU/CUDA through transformers. "onnx-dml" runs
    #: generation on the Radeon via onnxruntime-genai (30x faster here) and keeps
    #: hidden-state pooling in torch on CPU. See airComp/agents/llm_onnx.py.
    backend: str = "torch"
    device: str = "cuda"
    dtype: str = "float16"
    cache_dir: str = ".hf_cache"
    genai_dir: str = "onnx/qwen-genai-int4-dml"
    max_new_tokens: int = 200
    temperature: float = 0.7


@dataclass
class NegotiationConfig:
    item_types: tuple = ITEM_TYPES
    total_items_choices: tuple = (5, 6, 7)
    min_per_type: int = 1
    max_per_type: int = 4
    max_value_per_type: float = 10.0
    pool_value_points: float = 100.0
    max_messages: int = 10
    max_retries: int = 2
    include_rationale: bool = True
    #: A message the receiver cannot decode ends the episode as a no-deal.
    #:
    #: This is a structural asymmetry between the pipelines, not a property of
    #: either channel: a digital frame can be lost, so a digital agent gets one
    #: shot, while `SemanticDecoder` always emits a valid offer and therefore
    #: keeps all `max_messages` turns to converge. Set False to let the
    #: negotiation survive a lost message -- `history_prompt` already renders it
    #: as "[message lost / unparseable]" -- and the two pipelines then get the
    #: same number of attempts. True reproduces the Phase 3 results.
    lost_message_ends_episode: bool = True


@dataclass
class ChannelConfig:
    channel_mode: Literal["raw", "arq"] = "raw"
    snr_db: float = 10.0


@dataclass
class JSCCConfig:
    k: int = 16
    encoder_hidden_dims: tuple = (256, 128)
    decoder_hidden_dims: tuple = (128, 256)
    max_count: int = 4
    aux_dim: int = 1
    snr_range: tuple = (-5.0, 20.0)


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64
    action_loss_weight: float = 1.0
    aux_loss_weight: float = 0.1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    negotiation: NegotiationConfig = field(default_factory=NegotiationConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    jscc: JSCCConfig = field(default_factory=JSCCConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for section_name in ("model", "negotiation", "channel", "jscc", "train"):
        section = raw.get(section_name)
        if not section:
            continue
        current = getattr(cfg, section_name)
        for key, value in section.items():
            if isinstance(value, list):
                value = tuple(value)
            setattr(current, key, value)
    return cfg
