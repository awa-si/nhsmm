from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM


def _make_model(*, causal: bool = True) -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=causal,
        dropout=0.0,
        seed=7,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_general_functional_contracts() -> None:
    torch.manual_seed(7)
    model = _make_model(causal=True)

    assert model.device.type == "cpu"

    base_encoder = model.encoder.encoder
    assert getattr(base_encoder, "causal", False) is True
    assert getattr(base_encoder, "bidirectional", True) is False

    x = torch.randn(1, 8, model.config.n_features)

    with torch.inference_mode():
        encoded = base_encoder(x)
    assert encoded.shape[:2] == x.shape[:2]

    prefix_len = 5
    changed_future = x.clone()
    changed_future[:, prefix_len:] = torch.randn_like(changed_future[:, prefix_len:]) * 100.0

    with torch.inference_mode():
        prefix_a = base_encoder(x)[:, :prefix_len]
        prefix_b = base_encoder(changed_future)[:, :prefix_len]

    assert torch.allclose(prefix_a, prefix_b, atol=1e-6, rtol=1e-6)

    with torch.inference_mode():
        seq_set = model._build_sequence_set(x)
        alpha = model.forward(seq_set)

    assert torch.isfinite(alpha[:, 0, :, 0]).all()
    assert torch.isneginf(alpha[:, 0, :, 1:]).all()
