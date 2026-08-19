"""Thin wrapper around a local Hugging Face instruct model.

`chat_with_hidden` is the piece the semantic/JSCC pipeline depends on: it
returns not just the generated text but the mean-pooled last-layer hidden
state over the generated (offer-JSON) token span, which is what
`SemanticEncoder` compresses and sends over the simulated channel.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        cache_dir: str = ".hf_cache",
    ):
        self.model_name = model_name
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        # float32 on CPU, deliberately. bfloat16 halves the weight traffic and does speed
        # up decode, but this CPU is AVX2-only (no AVX512-BF16), so bf16 GEMM is emulated
        # and the compute-bound prefill in chat_with_hidden goes from 2.8 s to 17 s.
        # Measured net effect of bf16 on a full episode: 32 s -> 60 s. Do not "optimize"
        # this without re-measuring the prefill, not just tokens/s.
        torch_dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, dtype=torch_dtype
        ).to(self.device)
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _build_messages(self, system_prompt: str, history: list, user_prompt: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @torch.no_grad()
    def chat(
        self,
        system_prompt: str,
        history: list,
        user_prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        text, _ = self._generate(system_prompt, history, user_prompt, max_new_tokens, temperature, need_hidden=False)
        return text

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        """Mean-pooled input embedding of `text` -- an embedding-matrix lookup,
        not a forward pass. This is the space `SemanticDecoder`'s optional embed
        head is trained toward (`JsccExample.embed_target`,
        `airComp/jscc/dataset.py:offer_canonical_text`), so a soft prompt built
        from it lands where the model already expects text to live.
        """
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)["input_ids"]
        embeds = self.model.get_input_embeddings()(ids)[0]  # (seq_len, hidden_dim)
        return embeds.mean(dim=0).float().cpu()

    @torch.no_grad()
    def chat_with_soft_prompt(
        self,
        system_prompt: str,
        history: list,
        user_prompt: str,
        soft_prompt_embed: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Like `chat`, but `soft_prompt_embed` (hidden_dim,) is prepended as one
        extra embedding-space token in front of the tokenized prompt.

        This is the injection point the ONNX genai/DirectML backend cannot
        offer (it only ever calls `append_tokens` on token ids -- see
        CLAUDE.md), so it exists only here, on the CPU torch path.
        """
        messages = self._build_messages(system_prompt, history, user_prompt)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        text_embeds = self.model.get_input_embeddings()(inputs["input_ids"])  # (1, seq_len, hidden_dim)

        soft = soft_prompt_embed.to(device=self.device, dtype=text_embeds.dtype).view(1, 1, -1)
        inputs_embeds = torch.cat([soft, text_embeds], dim=1)
        attention_mask = torch.cat(
            [inputs["attention_mask"].new_ones((1, 1)), inputs["attention_mask"]], dim=1
        )

        gen_out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # generate() given inputs_embeds (no input_ids) returns ONLY the newly
        # generated ids -- there is no token-id prompt to slice past.
        return self.tokenizer.decode(gen_out[0], skip_special_tokens=True)

    @torch.no_grad()
    def score_completion_with_soft_prompt(
        self,
        system_prompt: str,
        history: list,
        user_prompt: str,
        soft_prompt_embed: torch.Tensor,
        completion_text: str,
    ) -> float:
        """Mean log-likelihood per token of `completion_text`, conditioned on
        `soft_prompt_embed` the same way `chat_with_soft_prompt` injects it.

        One forward pass, no generation loop -- this is the sensitive way to
        ask "does conditioning on this vector shift the model toward the
        right answer", without requiring the model to spontaneously generate
        well-formed output from a soft prompt it was never trained to
        interpret (see airComp/eval/injection_check.py).
        """
        messages = self._build_messages(system_prompt, history, user_prompt)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)["input_ids"]
        completion_ids = self.tokenizer(completion_text, return_tensors="pt").to(self.device)["input_ids"]

        embed = self.model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)  # (1, p, hidden)
        completion_embeds = embed(completion_ids)  # (1, c, hidden)
        soft = soft_prompt_embed.to(device=self.device, dtype=prompt_embeds.dtype).view(1, 1, -1)
        inputs_embeds = torch.cat([soft, prompt_embeds, completion_embeds], dim=1)

        ignore = torch.full((1, 1 + prompt_ids.shape[1]), -100, dtype=torch.long, device=self.device)
        labels = torch.cat([ignore, completion_ids], dim=1)

        loss = self.model(inputs_embeds=inputs_embeds, labels=labels).loss
        return -loss.item()

    @torch.no_grad()
    def chat_with_hidden(
        self,
        system_prompt: str,
        history: list,
        user_prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ):
        return self._generate(system_prompt, history, user_prompt, max_new_tokens, temperature, need_hidden=True)

    def _generate(self, system_prompt, history, user_prompt, max_new_tokens, temperature, need_hidden):
        messages = self._build_messages(system_prompt, history, user_prompt)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_ids = gen_out[0]
        gen_ids = full_ids[prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        hidden = None
        if need_hidden:
            out = self.model(full_ids.unsqueeze(0), output_hidden_states=True)
            last_layer = out.hidden_states[-1][0]  # (seq_len, hidden_dim)
            offer_span = last_layer[prompt_len:]
            if offer_span.shape[0] == 0:
                offer_span = last_layer[-1:]
            hidden = offer_span.mean(dim=0).float().cpu()
        return text, hidden
