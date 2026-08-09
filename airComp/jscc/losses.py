"""Loss functions for JSCC encoder/decoder training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def offer_ce_loss(offer_logits: torch.Tensor, true_counts: torch.Tensor) -> torch.Tensor:
    """offer_logits: (B, num_types, max_count+1); true_counts: (B, num_types) long."""
    num_types = offer_logits.shape[1]
    total = offer_logits.new_zeros(())
    for i in range(num_types):
        total = total + F.cross_entropy(offer_logits[:, i, :], true_counts[:, i])
    return total / num_types


def action_ce_loss(action_logits: torch.Tensor, true_action: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(action_logits, true_action)


def aux_mse_loss(aux_pred: torch.Tensor, aux_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(aux_pred, aux_true)


def expected_utility_loss(offer_logits: torch.Tensor, values_per_type: torch.Tensor, max_count: int) -> torch.Tensor:
    """Phase-2 differentiable surrogate: negative expected utility under the softmax
    distribution over each type's count, summed across types.

    values_per_type: (B, num_types) per-unit value for each type.
    """
    probs = torch.softmax(offer_logits, dim=-1)  # (B, num_types, max_count+1)
    counts = torch.arange(max_count + 1, device=offer_logits.device, dtype=probs.dtype)
    expected_counts = (probs * counts).sum(dim=-1)  # (B, num_types)
    expected_utility = (expected_counts * values_per_type).sum(dim=-1)  # (B,)
    return -expected_utility.mean()
