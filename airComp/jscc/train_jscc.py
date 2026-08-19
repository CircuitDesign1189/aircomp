"""Phase 1 training loop: supervised encoder+decoder training (through the
differentiable analog channel) on a frozen-LLM self-play dataset, with
SNR randomized per batch so the JSCC generalizes across channel conditions.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from airComp.channel.analog import AnalogAWGNChannel
from airComp.config import ITEM_TYPES, JSCCConfig, TrainConfig
from airComp.jscc.dataset import JsccExample, load_dataset
from airComp.jscc.losses import action_ce_loss, aux_mse_loss, embed_cosine_loss, expected_utility_loss, offer_ce_loss
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder, pool_to_mask


class _JsccTorchDataset(Dataset):
    def __init__(self, examples: list, item_types=ITEM_TYPES, max_count: int = 4,
                 include_values: bool = False, include_embed_target: bool = False):
        self.examples = examples
        self.item_types = item_types
        self.max_count = max_count
        self.include_values = include_values
        self.include_embed_target = include_embed_target

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex: JsccExample = self.examples[idx]
        counts = torch.tensor(
            [min(ex.counts.get(t, 0), self.max_count) for t in self.item_types], dtype=torch.long
        )
        mask = pool_to_mask(ex.pool, self.item_types, self.max_count)
        item = {
            "hidden": ex.hidden.float(),
            "counts": counts,
            "action": torch.tensor(ex.action_idx, dtype=torch.long),
            "aux": torch.tensor([ex.aux], dtype=torch.float32),
            "mask": mask,
        }
        if self.include_values:
            item["values"] = torch.tensor([ex.values.get(t, 0.0) for t in self.item_types], dtype=torch.float32)
        if self.include_embed_target:
            item["embed_target"] = ex.embed_target.float()
        return item


def train(
    dataset_path: str,
    out_ckpt: str,
    input_dim: int,
    jscc_cfg: JSCCConfig = JSCCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    device: str = "cpu",
    init_checkpoint: str | None = None,
) -> dict:
    """Phase 1 is `utility_loss_weight=0, init_checkpoint=None` -- unchanged.

    Phase 2 (single-turn utility fine-tuning, see docs/results.md's roadmap) is
    the same loop with `utility_loss_weight > 0`, typically continuing from a
    Phase-1 checkpoint via `init_checkpoint` rather than training from scratch.

    The injectable-embedding experiment (`SemanticDecoder`'s optional embed
    head) is `jscc_cfg.embed_dim` set + `embed_loss_weight > 0`; it needs a
    dataset collected with a backend that implements `embed_text` (CPU torch
    `LocalLLM` only) so every example carries `embed_target`.
    """
    examples = load_dataset(dataset_path)
    if not examples:
        raise ValueError(f"No examples found in {dataset_path}")

    want_embed = jscc_cfg.embed_dim is not None or train_cfg.embed_loss_weight > 0
    if want_embed and any(ex.embed_target is None for ex in examples):
        raise ValueError(
            "jscc_cfg.embed_dim/train_cfg.embed_loss_weight requested but this dataset has "
            "examples with no embed_target -- recollect it with a LocalLLM backend that "
            "implements embed_text (only the CPU torch backend does; see airComp/agents/llm_backend.py)"
        )
    want_utility = train_cfg.utility_loss_weight > 0
    if want_utility and any(not hasattr(ex, "values") for ex in examples):
        raise ValueError(
            "train_cfg.utility_loss_weight requested but this dataset has examples with no "
            "values field -- it predates that field, so Phase-2 utility training needs a "
            "fresh collect-dataset run (see airComp/jscc/dataset.py:JsccExample)"
        )

    torch_ds = _JsccTorchDataset(examples, max_count=jscc_cfg.max_count,
                                 include_values=want_utility, include_embed_target=want_embed)
    loader = DataLoader(torch_ds, batch_size=train_cfg.batch_size, shuffle=True)

    encoder = SemanticEncoder(input_dim, jscc_cfg.encoder_hidden_dims, jscc_cfg.k).to(device)
    decoder = SemanticDecoder(
        jscc_cfg.k, jscc_cfg.decoder_hidden_dims, len(ITEM_TYPES), jscc_cfg.max_count, jscc_cfg.aux_dim,
        embed_dim=jscc_cfg.embed_dim,
    ).to(device)
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        # strict=False: a Phase-1 checkpoint has no embed_head, since it was
        # trained with embed_dim=None. Growing into the embed-head experiment
        # from an existing checkpoint is the expected use of init_checkpoint,
        # not a mismatch -- report what was newly initialized rather than fail.
        missing, unexpected = decoder.load_state_dict(ckpt["decoder"], strict=False)
        if missing or unexpected:
            print(f"init_checkpoint: decoder missing {missing}, unexpected {unexpected}")
    channel = AnalogAWGNChannel().to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=train_cfg.lr)

    loss_history = []
    for _epoch in range(train_cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            hidden = batch["hidden"].to(device)
            counts = batch["counts"].to(device)
            action = batch["action"].to(device)
            aux = batch["aux"].to(device)
            mask = batch["mask"].to(device)

            snr_db = random.uniform(*jscc_cfg.snr_range)

            z = encoder(hidden)
            y = channel(z, snr_db)
            out = decoder(y, mask)

            loss = (
                offer_ce_loss(out["offer_logits"], counts)
                + train_cfg.action_loss_weight * action_ce_loss(out["action_logits"], action)
                + train_cfg.aux_loss_weight * aux_mse_loss(out["aux"], aux)
            )
            if train_cfg.utility_loss_weight:
                values = batch["values"].to(device)
                loss = loss + train_cfg.utility_loss_weight * expected_utility_loss(
                    out["offer_logits"], values, jscc_cfg.max_count
                )
            if train_cfg.embed_loss_weight and "embed" in out:
                embed_target = batch["embed_target"].to(device)
                loss = loss + train_cfg.embed_loss_weight * embed_cosine_loss(out["embed"], embed_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

    # Create the directory here, not in the caller: losing a finished training run
    # on its very last line because a folder was missing is the worst possible
    # failure mode for a job that takes minutes to hours.
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "input_dim": input_dim,
            "jscc_cfg": jscc_cfg,
        },
        out_ckpt,
    )
    return {"loss_history": loss_history}
