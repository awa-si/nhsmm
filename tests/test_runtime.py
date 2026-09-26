from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import HSMMFilterRuntime


def _make_model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=29,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_runtime_matches_model_bound_batch_filter() -> None:
    torch.manual_seed(29)
    model = _make_model()
    x = torch.randn(1, 7, model.config.n_features)

    batch = filter_model_sequence(model, x)
    runtime = HSMMFilterRuntime(model)

    streamed = []
    for t in range(x.shape[1]):
        state = runtime.step(x[:, t], timestamp=t)
        streamed.append(state.log_posterior)

    streamed_trace = torch.stack(streamed, dim=1)
    assert torch.allclose(
        streamed_trace,
        batch.log_posterior,
        atol=1e-6,
        rtol=1e-6,
    )


def test_duplicate_timestamp_does_not_advance_runtime() -> None:
    torch.manual_seed(31)
    model = _make_model()
    runtime = HSMMFilterRuntime(model)

    runtime.step(torch.randn(1, model.config.n_features), timestamp=100)
    before = runtime.state
    assert before is not None

    try:
        runtime.step(torch.randn(1, model.config.n_features), timestamp=100)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("duplicate timestamps must be rejected")

    assert runtime.state is before


def test_timestamp_mode_is_fixed_until_reset() -> None:
    model = _make_model()
    runtime = HSMMFilterRuntime(model)

    runtime.step(torch.randn(1, model.config.n_features))
    try:
        runtime.step(torch.randn(1, model.config.n_features), timestamp=1)
    except ValueError as exc:
        assert "fixed at initialization" in str(exc)
    else:
        raise AssertionError("timestamp mode must not change midstream")

    runtime.reset()
    runtime.step(torch.randn(1, model.config.n_features), timestamp=1)
    assert runtime.state is not None
    assert runtime.state.uses_timestamps is True
