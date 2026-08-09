"""Semantic encoder/decoder for the JSCC pipeline.

SemanticEncoder compresses an LLM hidden state into a k-dim, power-normalized
real vector (the "channel uses" budget). SemanticDecoder reconstructs a
structured offer directly (item counts + action + an auxiliary continuous
"intent" signal) rather than attempting to reconstruct injectable LLM context
across independently-instantiated model calls -- see CLAUDE.md for why.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from airComp.env.negotiation import Pool


class SemanticEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128), k: int = 16):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.LayerNorm(h1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, k),
        )
        self.k = k

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (self.k**0.5) * z / norm


class SemanticDecoder(nn.Module):
    def __init__(self, k: int, hidden_dims=(128, 256), num_types: int = 3, max_count: int = 4, aux_dim: int = 1):
        super().__init__()
        h1, h2 = hidden_dims
        self.trunk = nn.Sequential(
            nn.Linear(k, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.num_types = num_types
        self.max_count = max_count
        self.offer_head = nn.Linear(h2, num_types * (max_count + 1))
        self.action_head = nn.Linear(h2, 3)
        self.aux_head = nn.Linear(h2, aux_dim)

    def forward(self, y: torch.Tensor, pool_mask: torch.Tensor) -> dict:
        """pool_mask: (batch, num_types, max_count+1) bool -- True where a count is
        feasible (<= pool count for that type). Infeasible logits are masked to -inf
        identically at train and inference time, so decoded offers are always feasible."""
        features = self.trunk(y)
        offer_logits = self.offer_head(features).view(-1, self.num_types, self.max_count + 1)
        offer_logits = offer_logits.masked_fill(~pool_mask, float("-inf"))
        action_logits = self.action_head(features)
        aux = self.aux_head(features)
        return {"offer_logits": offer_logits, "action_logits": action_logits, "aux": aux}


def pool_to_mask(pool: Pool, item_types, max_count: int) -> torch.Tensor:
    mask = torch.zeros(len(item_types), max_count + 1, dtype=torch.bool)
    for i, t in enumerate(item_types):
        limit = min(pool.counts[t], max_count)
        mask[i, : limit + 1] = True
    return mask
