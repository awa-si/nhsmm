from __future__ import annotations

import pytest
import torch

from nhsmm.encoder import DefaultEncoder


@pytest.mark.parametrize("kernel", [2, 3, 4, 5])
def test_noncausal_encoder_preserves_sequence_length_for_any_kernel(kernel: int) -> None:
    encoder = DefaultEncoder(
        n_features=3,
        cnn_kernel=kernel,
        hidden_dim=4,
        cnn_channels=5,
        bidirectional=False,
        dropout=0.0,
        causal=False,
    ).eval()
    x = torch.randn(2, 7, 3)

    out = encoder(x)

    assert out.shape == (2, 7, 4)


def test_encoder_rejects_non_prefix_mask() -> None:
    encoder = DefaultEncoder(
        n_features=3,
        hidden_dim=4,
        cnn_channels=5,
        bidirectional=False,
        dropout=0.0,
        causal=True,
    )
    x = torch.randn(1, 5, 3)
    mask = torch.tensor([[True, False, True, False, False]])

    with pytest.raises(ValueError, match="right-padded"):
        encoder(x, mask=mask)


def test_packed_encoder_handles_empty_mask_rows_without_fake_timestep() -> None:
    encoder = DefaultEncoder(
        n_features=3,
        hidden_dim=4,
        cnn_channels=5,
        bidirectional=False,
        dropout=0.0,
        causal=True,
        use_packed=True,
        return_sequence=False,
    ).eval()
    x = torch.randn(2, 5, 3)
    mask = torch.tensor([[False, False, False, False, False], [True, True, False, False, False]])

    last = encoder(x, mask=mask, return_sequence=False)
    sequence = encoder(x, mask=mask, return_sequence=True)

    torch.testing.assert_close(last[0], torch.zeros_like(last[0]))
    torch.testing.assert_close(sequence[0], torch.zeros_like(sequence[0]))
    torch.testing.assert_close(sequence[1, 2:], torch.zeros_like(sequence[1, 2:]))


def test_encoder_rejects_mask_shape_mismatch() -> None:
    encoder = DefaultEncoder(
        n_features=3,
        hidden_dim=4,
        cnn_channels=5,
        bidirectional=False,
        dropout=0.0,
        causal=True,
    )
    x = torch.randn(2, 5, 3)

    with pytest.raises(ValueError, match="does not match"):
        encoder(x, mask=torch.ones(2, 4, dtype=torch.bool))


def test_context_cache_does_not_retain_autograd_graph() -> None:
    encoder = DefaultEncoder(
        n_features=3,
        hidden_dim=4,
        cnn_channels=5,
        bidirectional=False,
        dropout=0.0,
        causal=True,
    )
    x = torch.randn(2, 5, 3, requires_grad=True)

    out = encoder(x)

    assert out.grad_fn is not None
    assert encoder._context is not None
    assert encoder._context.grad_fn is None


def test_noncausal_masked_padding_cannot_change_valid_outputs() -> None:
    torch.manual_seed(3)
    encoder = DefaultEncoder(
        n_features=2,
        cnn_kernel=3,
        hidden_dim=4,
        cnn_channels=5,
        bidirectional=False,
        dropout=0.0,
        causal=False,
        use_packed=True,
    ).eval()
    baseline = torch.randn(1, 5, 2)
    perturbed = baseline.clone()
    perturbed[:, 3:] = 1000.0
    mask = torch.tensor([[True, True, True, False, False]])

    baseline_out = encoder(baseline, mask=mask)
    perturbed_out = encoder(perturbed, mask=mask)

    torch.testing.assert_close(baseline_out[:, :3], perturbed_out[:, :3])


def test_causal_encoder_is_prefix_invariant() -> None:
    torch.manual_seed(7)
    encoder = DefaultEncoder(
        n_features=4,
        hidden_dim=4,
        cnn_channels=8,
        cnn_kernel=3,
        bidirectional=False,
        dropout=0.0,
        causal=True,
    ).eval()
    x = torch.randn(1, 8, 4)
    prefix_len = 5
    changed_future = x.clone()
    changed_future[:, prefix_len:] = torch.randn_like(changed_future[:, prefix_len:]) * 100.0

    with torch.inference_mode():
        prefix_a = encoder(x)[:, :prefix_len]
        prefix_b = encoder(changed_future)[:, :prefix_len]

    torch.testing.assert_close(prefix_a, prefix_b, atol=1e-6, rtol=1e-6)
