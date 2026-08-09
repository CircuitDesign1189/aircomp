"""Phase 1 training loop: supervised encoder+decoder training (through the
differentiable analog channel) on a frozen-LLM self-play dataset, with
SNR randomized per batch so the JSCC generalizes across channel conditions.
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset

from airComp.channel.analog import AnalogAWGNChannel
from airComp.config import ITEM_TYPES, JSCCConfig, TrainConfig
from airComp.jscc.dataset import JsccExample, load_dataset
from airComp.jscc.losses import action_ce_loss, aux_mse_loss, offer_ce_loss
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder, pool_to_mask


class _JsccTorchDataset(Dataset):
    def __init__(self, examples: list, item_types=ITEM_TYPES, max_count: int = 4):
        self.examples = examples
        self.item_types = item_types
        self.max_count = max_count

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex: JsccExample = self.examples[idx]
        counts = torch.tensor(
            [min(ex.counts.get(t, 0), self.max_count) for t in self.item_types], dtype=torch.long
        )
        mask = pool_to_mask(ex.pool, self.item_types, self.max_count)
        return {
            "hidden": ex.hidden.float(),
            "counts": counts,
            "action": torch.tensor(ex.action_idx, dtype=torch.long),
            "aux": torch.tensor([ex.aux], dtype=torch.float32),
            "mask": mask,
        }


def train(
    dataset_path: str,
    out_ckpt: str,
    input_dim: int,
    jscc_cfg: JSCCConfig = JSCCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    device: str = "cpu",
) -> dict:
    examples = load_dataset(dataset_path)
    if not examples:
        raise ValueError(f"No examples found in {dataset_path}")

    torch_ds = _JsccTorchDataset(examples, max_count=jscc_cfg.max_count)
    loader = DataLoader(torch_ds, batch_size=train_cfg.batch_size, shuffle=True)

    encoder = SemanticEncoder(input_dim, jscc_cfg.encoder_hidden_dims, jscc_cfg.k).to(device)
    decoder = SemanticDecoder(
        jscc_cfg.k, jscc_cfg.decoder_hidden_dims, len(ITEM_TYPES), jscc_cfg.max_count, jscc_cfg.aux_dim
    ).to(device)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

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
