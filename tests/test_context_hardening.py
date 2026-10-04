from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet
from nhsmm.encoder import DefaultEncoder


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


class _MaskAwareEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x, mask=None):
        self.mask = mask
        return x


def test_context_encoder_forwards_supported_mask():
    inner = _MaskAwareEncoder()
    encoder = ContextEncoder(inner, layer_norm=False)
    mask = torch.tensor([[True, True, False]])

    encoder(torch.ones(1, 3, 1), mask=mask)

    assert inner.mask is not None
    assert torch.equal(inner.mask, mask)


def test_sequence_update_preserves_temporal_context_shape():
    sequence = SequenceSet.from_unbatched([torch.tensor([[1.0], [3.0]]), torch.tensor([[10.0]])])
    encoder = ContextEncoder(_IdentityEncoder(), pool="mean", layer_norm=False)

    sequence.update(encoder)

    assert sequence.contexts.shape == sequence.sequences.shape
    assert sequence.canonical.shape == (2, 1, 1)
    assert torch.equal(sequence.contexts[1, :1], torch.tensor([[10.0]]))
    assert sequence.contexts[1, 1:].eq(0).all()


def test_sequence_set_rejects_incomplete_parallel_inputs():
    sequences = [torch.ones(2, 1), torch.ones(2, 1)]

    with pytest.raises(ValueError, match="contexts length"):
        SequenceSet.from_unbatched(sequences, contexts=[torch.ones(2, 1)])

    with pytest.raises(ValueError, match="log_probs length"):
        SequenceSet.from_unbatched(sequences, log_probs=[torch.ones(2, 2)])


def test_sequence_set_rejects_log_prob_time_mismatch():
    with pytest.raises(ValueError, match="log_probs length"):
        SequenceSet.from_unbatched(
            [torch.ones(2, 1)],
            log_probs=[torch.ones(3, 2)],
        )


def test_context_router_rejects_log_prob_time_mismatch():
    with pytest.raises(ValueError, match="batch/time"):
        ContextRouter(
            context=torch.zeros(2, 3, 4),
            canonical=torch.zeros(2, 1, 4),
            log_probs=torch.zeros(2, 99, 5),
        )


def test_prepare_mask_normalizes_device_and_dtype():
    encoder = ContextEncoder(_IdentityEncoder())
    x = torch.ones(1, 3, 1)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)

    prepared = encoder._prepare_mask(mask, 1, 3, device=x.device)

    assert prepared.dtype == torch.bool
    assert prepared.device == x.device


def test_context_encoder_caches_do_not_retain_autograd_graph() -> None:
    torch.manual_seed(71)
    raw = DefaultEncoder(
        n_features=3,
        hidden_dim=4,
        cnn_channels=5,
        cnn_kernel=3,
        bidirectional=False,
        dropout=0.0,
        causal=True,
    )
    encoder = ContextEncoder(raw, pool="mean")
    x = torch.randn(2, 5, 3, requires_grad=True)

    sequence, canonical, _ = encoder(
        x,
        return_sequence=True,
        return_context=True,
    )

    assert sequence.grad_fn is not None
    assert canonical is not None
    assert canonical.grad_fn is not None
    assert encoder._sequence is not None
    assert encoder._context is not None
    assert encoder._sequence.grad_fn is None
    assert encoder._context.grad_fn is None

    (sequence.sum() + canonical.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_context_encoder_reset_clears_only_transient_detached_caches() -> None:
    torch.manual_seed(73)
    raw = DefaultEncoder(
        n_features=3,
        hidden_dim=4,
        cnn_channels=5,
        cnn_kernel=3,
        bidirectional=False,
        dropout=0.0,
        causal=True,
    )
    encoder = ContextEncoder(raw, pool="attn")
    x = torch.randn(2, 5, 3)

    encoder(x, return_sequence=True, return_context=True)
    attn_parameter = encoder._attn_vector

    assert encoder._sequence is not None
    assert encoder._context is not None
    assert attn_parameter is not None

    encoder.reset()

    assert encoder._sequence is None
    assert encoder._context is None
    assert encoder._attn_vector is attn_parameter
