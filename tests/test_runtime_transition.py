from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.runtime import HSMMFilterRuntime


def _make_model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=47,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_runtime_transition_forecast_contract() -> None:
    torch.manual_seed(47)
    model = _make_model()
    runtime = HSMMFilterRuntime(model)
    runtime.step(torch.randn(1, model.config.n_features), timestamp=1)

    forecast = runtime.forecast_transition()

    assert forecast.boundary_transition_joint.shape == (1, model.config.n_states, model.config.n_states)
    assert forecast.next_episode_state_joint.shape == (1, model.config.n_states)
    assert forecast.next_state_prior.shape == (1, model.config.n_states)
    assert forecast.episode_end_probability.shape == (1,)
    assert forecast.state_change_probability.shape == (1,)
    assert torch.allclose(forecast.next_state_prior.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=0.0)
    assert torch.allclose(
        forecast.boundary_transition_joint.sum(dim=(1, 2)),
        forecast.episode_end_probability,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.all(forecast.state_change_probability <= forecast.episode_end_probability + 1e-6)


def test_runtime_transition_forecast_requires_state() -> None:
    runtime = HSMMFilterRuntime(_make_model())

    try:
        runtime.forecast_transition()
    except RuntimeError as exc:
        assert "at least one observation" in str(exc)
    else:
        raise AssertionError("transition forecast must require initialized runtime state")
