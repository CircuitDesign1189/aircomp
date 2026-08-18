"""The backend switch is the one place that decides which engine every run uses.

Before the factory existed, seven call sites each constructed LocalLLM directly,
so a half-finished switch would silently leave some runs on a different numeric
path -- and for the semantic pipeline that means hidden states the JSCC decoder
was never trained on. These tests need no model weights: they pin the wiring.
"""
from __future__ import annotations

import pytest

from airComp.agents.factory import build_llm
from airComp.config import ModelConfig


def test_unknown_backend_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown model.backend 'gpu'"):
        build_llm(ModelConfig(backend="gpu"))


def test_default_backend_is_torch():
    """Torch stays the default so a checkout without the built genai model works."""
    assert ModelConfig().backend == "torch"


def test_onnx_backend_says_how_to_build_the_model_when_it_is_missing(tmp_path):
    """The genai directory is a build artifact, not something a clone provides."""
    from airComp.agents.llm_onnx import OnnxDmlLLM

    with pytest.raises(FileNotFoundError, match="build_genai_model"):
        OnnxDmlLLM(genai_dir=str(tmp_path / "absent"))


def test_shipped_configs_select_the_gpu_backend():
    """These are the configs the long runs actually use; a silent revert to torch
    would cost hours per collection."""
    from airComp.config import load_config

    for path in ("configs/base.yaml", "configs/snr_sweep.yaml"):
        assert load_config(path).model.backend == "onnx-dml", path
