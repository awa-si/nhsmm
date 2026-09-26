from __future__ import annotations

import torch

from nhsmm.distributions.default import Categorical


def test_categorical_rsample_soft_and_hard() -> None:
    torch.manual_seed(17)
    logits = torch.randn(3, 4, requires_grad=True)
    dist = Categorical(logits=logits)

    soft = dist.rsample(temperature=0.7)
    assert soft.shape == logits.shape
    assert torch.isfinite(soft).all()
    assert torch.allclose(soft.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)

    soft.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    hard = dist.rsample(temperature=0.7, hard=True)
    assert hard.shape == logits.shape
    assert torch.all(hard.sum(dim=-1) == 1)
    assert torch.all((hard == 0) | (hard == 1))
