# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""One place that decides which LLM backend the whole project uses.

Before this, seven call sites each constructed LocalLLM directly, so switching
backends meant editing seven files -- and any one of them left behind would
silently run a different engine, which for the semantic pipeline means hidden
states from a different numeric path.
"""
from __future__ import annotations

from airComp.config import ModelConfig


def build_llm(cfg: ModelConfig):
    """Construct the configured LLM backend. Both satisfy the same interface:
    `chat`, `chat_with_hidden`, `hidden_size`."""
    if cfg.backend == "onnx-dml":
        from airComp.agents.llm_onnx import OnnxDmlLLM

        return OnnxDmlLLM(cfg.model_name, cfg.device, cfg.dtype, cfg.cache_dir, cfg.genai_dir)
    if cfg.backend != "torch":
        raise ValueError(f"unknown model.backend {cfg.backend!r} (expected 'torch' or 'onnx-dml')")

    from airComp.agents.llm_backend import LocalLLM

    return LocalLLM(cfg.model_name, cfg.device, cfg.dtype, cfg.cache_dir)
