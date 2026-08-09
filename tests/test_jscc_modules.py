import torch

from airComp.config import ITEM_TYPES
from airComp.env.negotiation import Pool
from airComp.jscc.modules import SemanticDecoder, SemanticEncoder, pool_to_mask


def test_encoder_output_is_power_normalized():
    k = 16
    encoder = SemanticEncoder(input_dim=32, k=k)
    h = torch.randn(5, 32)
    z = encoder(h)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, k**0.5), atol=1e-3)


def test_pool_to_mask_matches_pool_bounds():
    pool = Pool(counts={"book": 2, "hat": 4, "ball": 0})
    mask = pool_to_mask(pool, ITEM_TYPES, max_count=4)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True, True]
    assert mask[2].tolist() == [True, False, False, False, False]


def test_decoder_masks_infeasible_counts():
    k, max_count = 8, 4
    decoder = SemanticDecoder(k=k, num_types=len(ITEM_TYPES), max_count=max_count)
    pool = Pool(counts={"book": 1, "hat": 4, "ball": 2})
    mask = pool_to_mask(pool, ITEM_TYPES, max_count).unsqueeze(0)
    y = torch.randn(1, k)
    out = decoder(y, mask)
    probs = torch.softmax(out["offer_logits"], dim=-1)
    assert probs[0, 0, 2:].sum().item() < 1e-6  # book pool max is 1
    assert probs[0, 2, 3:].sum().item() < 1e-6  # ball pool max is 2
    assert out["action_logits"].shape == (1, 3)


def test_decoder_output_shapes_batch():
    k, max_count, batch = 8, 4, 3
    decoder = SemanticDecoder(k=k, num_types=len(ITEM_TYPES), max_count=max_count, aux_dim=1)
    pool = Pool(counts={"book": 2, "hat": 2, "ball": 2})
    mask = pool_to_mask(pool, ITEM_TYPES, max_count).unsqueeze(0).expand(batch, -1, -1)
    y = torch.randn(batch, k)
    out = decoder(y, mask)
    assert out["offer_logits"].shape == (batch, len(ITEM_TYPES), max_count + 1)
    assert out["action_logits"].shape == (batch, 3)
    assert out["aux"].shape == (batch, 1)
