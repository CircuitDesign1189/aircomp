import torch

from airComp.channel.analog import AnalogAWGNChannel


def test_output_shape_matches_input():
    channel = AnalogAWGNChannel()
    z = torch.randn(4, 16)
    y = channel(z, snr_db=10.0)
    assert y.shape == z.shape


def test_high_snr_is_approximately_lossless():
    channel = AnalogAWGNChannel()
    z = torch.ones(1, 16)
    y = channel(z, snr_db=80.0)
    assert torch.allclose(y, z, atol=0.05)


def test_gradients_flow_through_channel():
    channel = AnalogAWGNChannel()
    z = torch.randn(2, 8, requires_grad=True)
    y = channel(z, snr_db=5.0)
    loss = y.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.allclose(z.grad, torch.ones_like(z))
