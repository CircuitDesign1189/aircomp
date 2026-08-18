"""The LocalLLM interface, generated on the Radeon via onnxruntime-genai + DirectML.

Why this exists: torch has no GPU path on this machine (AMD card, so no CUDA; and
no torch-directml wheel for Python 3.14), but ONNX Runtime's DirectML provider does.

Measured on this bench, generating with a 250-token prompt:

    CPU torch fp32                     345 ms/token
    DirectML, generic ONNX graph       189 ms/token   (1.8x)
    DirectML, onnxruntime-genai         11 ms/token   (30x)

The gap between the last two is the whole reason this file uses genai rather than a
hand-written KV-cache loop over a plain exported graph. Decoding one token is a
batch-1, one-position pass: hundreds of tiny operators, so it is bound by per-op
dispatch overhead, not memory bandwidth. (Confirmed two ways: raising KV-cache
traffic 5x cost only 8%, and the int4 graph was *slower* than fp16 because
dequantization adds more ops.) genai ships fused decode kernels, which removes
that overhead; a generic graph cannot.

Only GENERATION runs on the GPU. `chat_with_hidden` still pools its hidden state
from the torch model on CPU, deliberately: that keeps the pooled vector
numerically identical to everything collected so far, so the JSCC decoder sees no
train/inference distribution shift. It is now the dominant cost per turn and is
the obvious next thing to move.

Build the model directory with `python scripts/build_genai_model.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

DEFAULT_GENAI_DIR = "onnx/qwen-genai-int4-dml"


class OnnxDmlLLM:
    """Drop-in for LocalLLM: same chat / chat_with_hidden / hidden_size surface."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "dml",  # accepted and ignored, for LocalLLM signature parity
        dtype: str = "int4",
        cache_dir: str = ".hf_cache",
        genai_dir: str = DEFAULT_GENAI_DIR,
    ):
        import onnxruntime_genai as og

        if not Path(genai_dir, "genai_config.json").exists():
            raise FileNotFoundError(
                f"no genai model at {genai_dir!r}. Build it first:\n"
                f"    python scripts/build_genai_model.py --out {genai_dir}"
            )
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "dml"
        self._og = og
        self.model = og.Model(genai_dir)

        # HF tokenizer for BOTH templating and encoding, even though genai ships its
        # own: the hidden-state pooling span is defined by a prompt-length token
        # count, so the ids fed to the GPU must be the same ids the torch model sees.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._torch_llm = None  # built on first chat_with_hidden; see module docstring

    @property
    def hidden_size(self) -> int:
        return self._torch_backend().hidden_size

    def _torch_backend(self):
        """The CPU torch model, loaded only if hidden states are actually wanted.

        A baseline-only run never calls chat_with_hidden, and skipping a 6 GB load
        for those runs is worth the lazy init.
        """
        if self._torch_llm is None:
            from airComp.agents.llm_backend import LocalLLM

            self._torch_llm = LocalLLM(self.model_name, "cpu", "float32", self.cache_dir)
        return self._torch_llm

    # -- prompt plumbing, identical to LocalLLM so the two stay comparable --------

    def _prompt_ids(self, system_prompt: str, history: list, user_prompt: str) -> np.ndarray:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)

    def _generate(self, prompt_ids: np.ndarray, max_new_tokens: int, temperature: float) -> list:
        ids = prompt_ids[0].tolist()
        params = self._og.GeneratorParams(self.model)
        if temperature > 0:
            params.set_search_options(
                max_length=len(ids) + max_new_tokens, do_sample=True, temperature=float(temperature)
            )
        else:
            params.set_search_options(max_length=len(ids) + max_new_tokens, do_sample=False)

        generator = self._og.Generator(self.model, params)
        generator.append_tokens(ids)
        produced = 0
        while not generator.is_done() and produced < max_new_tokens:
            generator.generate_next_token()
            produced += 1
        return list(generator.get_sequence(0))[len(ids) :]

    # -- LocalLLM interface -------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        history: list,
        user_prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        ids = self._prompt_ids(system_prompt, history, user_prompt)
        return self.tokenizer.decode(
            self._generate(ids, max_new_tokens, temperature), skip_special_tokens=True
        )

    def chat_with_hidden(
        self,
        system_prompt: str,
        history: list,
        user_prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ):
        """Generate on the GPU, then pool the hidden state on the CPU.

        The pooled span is the generated-token span, exactly as LocalLLM defines it.
        CLAUDE.md treats that span as load-bearing for the fairness of the two
        pipelines, so it is reproduced here rather than reinterpreted -- and the
        generated token ids are reused directly instead of re-tokenizing the decoded
        text, which would not reliably round-trip.
        """
        prompt_ids = self._prompt_ids(system_prompt, history, user_prompt)
        gen_ids = self._generate(prompt_ids, max_new_tokens, temperature)
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        full = np.concatenate(
            [prompt_ids, np.array(gen_ids, dtype=np.int64).reshape(1, -1)], axis=1
        )
        llm = self._torch_backend()
        with torch.no_grad():
            out = llm.model(torch.from_numpy(full), output_hidden_states=True)
        last_layer = out.hidden_states[-1][0]
        span = last_layer[prompt_ids.shape[1] :]
        if span.shape[0] == 0:
            span = last_layer[-1:]
        return text, span.mean(dim=0).float().cpu()
