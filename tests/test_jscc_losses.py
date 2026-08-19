import torch

from airComp.jscc.losses import embed_cosine_loss, expected_utility_loss


def test_embed_cosine_loss_is_zero_for_aligned_vectors():
    target = torch.randn(5, 16)
    pred = target * 3.0  # same direction, different norm -- must not matter

    loss = embed_cosine_loss(pred, target)

    assert loss.item() < 1e-5


def test_embed_cosine_loss_is_large_for_opposed_vectors():
    target = torch.randn(5, 16)
    pred = -target

    loss = embed_cosine_loss(pred, target)

    assert loss.item() > 1.9  # 1 - cos(180 deg) = 2


def test_embed_cosine_loss_is_differentiable():
    pred = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 8)

    embed_cosine_loss(pred, target).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_expected_utility_loss_prefers_higher_value_allocation():
    max_count = 4
    values = torch.tensor([[10.0, 1.0]])  # type 0 much more valuable than type 1

    logits_good = torch.zeros(1, 2, max_count + 1)
    logits_good[:, 0, max_count] = 10.0  # confidently take all of the valuable type
    logits_good[:, 1, 0] = 10.0          # confidently take none of the cheap type

    logits_bad = torch.zeros(1, 2, max_count + 1)
    logits_bad[:, 0, 0] = 10.0
    logits_bad[:, 1, max_count] = 10.0

    loss_good = expected_utility_loss(logits_good, values, max_count)
    loss_bad = expected_utility_loss(logits_bad, values, max_count)

    assert loss_good < loss_bad  # lower loss (= higher expected utility) for the better allocation


def test_expected_utility_loss_is_differentiable_through_logits():
    logits = torch.randn(4, 3, 5, requires_grad=True)
    values = torch.rand(4, 3)

    expected_utility_loss(logits, values, max_count=4).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
