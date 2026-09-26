from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence


def _make_model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=19,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_model_filter_trace_is_normalized() -> None:
    torch.manual_seed(19)
    model = _make_model()
    x = torch.randn(2, 7, model.config.n_features)

    trace = filter_model_sequence(model, x)

    assert trace.log_posterior.shape == (2, 7, 3, 5)
    log_z = trace.log_posterior.flatten(2).logsumexp(dim=-1)
    assert torch.allclose(log_z, torch.zeros_like(log_z), atol=1e-6, rtol=1e-6)


def test_model_filter_is_prefix_invariant() -> None:
    torch.manual_seed(23)
    model = _make_model()
    x = torch.randn(1, 8, model.config.n_features)
    prefix_len = 5

    changed_future = x.clone()
    changed_future[:, prefix_len:] = torch.randn_like(changed_future[:, prefix_len:]) * 100.0

    trace_a = filter_model_sequence(model, x)
    trace_b = filter_model_sequence(model, changed_future)

    assert torch.allclose(
        trace_a.log_posterior[:, :prefix_len],
        trace_b.log_posterior[:, :prefix_len],
        atol=1e-6,
        rtol=1e-6,
    )


def test_model_filter_requires_eval_and_causal_mode() -> None:
    model = _make_model()
    x = torch.randn(1, 3, model.config.n_features)

    model.train()
    try:
        filter_model_sequence(model, x)
    except RuntimeError as exc:
        assert "eval mode" in str(exc)
    else:
        raise AssertionError("training-mode filtering must fail closed")

    model.eval()
    model.config.causal = False
    try:
        filter_model_sequence(model, x)
    except ValueError as exc:
        assert "causal" in str(exc)
    else:
        raise AssertionError("non-causal model filtering must fail closed")
