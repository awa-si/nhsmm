from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nhsmm.context import ContextEncoder, SequenceSet


class _IdentityEncoder(nn.Module):
    def forward(self, x):
        return x


class _FailingEncoder(nn.Module):
    def forward(self, x):
        raise RuntimeError("boom")


def test_sequence_helpers_preserve_device() -> None:
    x1 = torch.randn(3, 2)
    x2 = torch.randn(2, 2)
    seq = SequenceSet.from_unbatched([x1, x2])
    assert seq.sequences.device == x1.device
    assert seq.lengths.device == x1.device
    assert seq.masks.device == x1.device
    assert seq.contexts.device == x1.device
    assert SequenceSet.batchify([x1, x2]).device == x1.device


def test_encode_restores_pool_after_failure() -> None:
    encoder = ContextEncoder(_FailingEncoder(), pool="mean")
    with pytest.raises(RuntimeError, match="boom"):
        encoder.encode(torch.randn(1, 3, 2), pool="max")
    assert encoder.pool == "mean"


def test_attention_pooling_cpu_smoke() -> None:
    x = torch.randn(2, 4, 6)
    encoder = ContextEncoder(_IdentityEncoder(), pool="attn")
    seq, canonical, _ = encoder(x, return_sequence=True, return_context=True)
    assert seq.device == x.device
    assert canonical is not None and canonical.device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("pool", ["mean", "attn", "mha"])
def test_context_encoder_cuda_device_safety(pool: str) -> None:
    device = torch.device("cuda")
    x = torch.randn(2, 4, 6, device=device)
    encoder = ContextEncoder(_IdentityEncoder(), pool=pool, n_heads=2).to(device)
    seq, canonical, _ = encoder(x, return_sequence=True, return_context=True)
    assert seq.device == device
    assert canonical is not None and canonical.device == device
