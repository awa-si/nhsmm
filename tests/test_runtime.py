from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence, next_episode_end_probability
from nhsmm.runtime import HSMMFilterRuntime, HSMMRuntimeState


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


def test_runtime_survival_forecast_uses_current_filter_state() -> None:
    torch.manual_seed(37)
    model = _make_model()
    runtime = HSMMFilterRuntime(model)

    try:
        runtime.forecast_survival((1, 3))
    except RuntimeError as exc:
        assert "at least one observation" in str(exc)
    else:
        raise AssertionError("forecasting before initialization must fail")

    runtime.step(torch.randn(1, model.config.n_features), timestamp=1)
    assert runtime.state is not None

    forecast = runtime.forecast_survival((1, 3, 6))
    one_step = next_episode_end_probability(
        runtime.state.filter_state,
        runtime.state.duration_log_prob,
    )

    assert forecast.horizons.tolist() == [1, 3, 6]
    assert forecast.survival_probability.shape == (1, 3)
    assert forecast.end_within_probability.shape == (1, 3)
    assert torch.allclose(
        forecast.end_within_probability[:, 0],
        one_step,
        atol=1e-6,
        rtol=1e-6,
    )


def test_runtime_matches_batch_filter_with_explicit_context_dim() -> None:
    torch.manual_seed(41)
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        context_dim=2,
        dropout=0.0,
        seed=41,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    x = torch.randn(2, 6, config.n_features)

    assert model.encoder.encoder.hidden_dim == 2
    batch = filter_model_sequence(model, x)
    runtime = HSMMFilterRuntime(model)
    streamed = [runtime.step(x[:, t]).log_posterior for t in range(x.shape[1])]

    torch.testing.assert_close(
        torch.stack(streamed, dim=1),
        batch.log_posterior,
        atol=1e-6,
        rtol=1e-6,
    )


def test_runtime_external_context_matches_batch_filter() -> None:
    torch.manual_seed(43)
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        context_dim=2,
        dropout=0.0,
        seed=43,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    x = torch.randn(2, 7, config.n_features)
    context = torch.randn(2, 7, 2)

    batch = filter_model_sequence(model, x, context=context)
    runtime = HSMMFilterRuntime(model)
    streamed = [
        runtime.step(x[:, t], context=context[:, t : t + 1]).log_posterior
        for t in range(x.shape[1])
    ]

    torch.testing.assert_close(
        torch.stack(streamed, dim=1),
        batch.log_posterior,
        atol=1e-6,
        rtol=1e-6,
    )


def test_runtime_external_context_mode_is_fixed_until_reset() -> None:
    model = _make_model()
    runtime = HSMMFilterRuntime(model)
    observation = torch.randn(1, model.config.n_features)
    context = torch.randn(model.context_dim)

    runtime.step(observation, context=context)
    try:
        runtime.step(observation)
    except ValueError as exc:
        assert "context mode is fixed" in str(exc)
    else:
        raise AssertionError("runtime context mode must not change midstream")

    runtime.reset()
    runtime.step(observation)
    try:
        runtime.step(observation, context=context)
    except ValueError as exc:
        assert "context mode is fixed" in str(exc)
    else:
        raise AssertionError("runtime context mode must not change midstream")


def test_runtime_state_rejects_nonfinite_observation() -> None:
    model = _make_model()
    runtime = HSMMFilterRuntime(model)
    runtime.step(torch.zeros(1, model.config.n_features))
    state = runtime.state
    assert state is not None

    bad = state.observations.clone()
    bad[..., 0] = float("nan")
    try:
        HSMMRuntimeState(
            filter_state=state.filter_state,
            observations=bad,
            duration_log_prob=state.duration_log_prob,
            transition_log_prob=state.transition_log_prob,
            encoder_state=state.encoder_state,
            last_timestamp=state.last_timestamp,
            uses_timestamps=state.uses_timestamps,
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite runtime observations must fail closed")


def test_runtime_rejects_nonfinite_observation_before_state_mutation() -> None:
    model = _make_model()
    runtime = HSMMFilterRuntime(model)
    observation = torch.zeros(1, model.config.n_features)
    observation[0, 0] = float("nan")

    try:
        runtime.step(observation)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite observations must fail closed")

    assert runtime.state is None


def test_runtime_rejects_nonfinite_external_context_before_state_mutation() -> None:
    model = _make_model()
    runtime = HSMMFilterRuntime(model)
    context = torch.zeros(model.context_dim)
    context[0] = float("inf")

    try:
        runtime.step(torch.zeros(1, model.config.n_features), context=context)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite external context must fail closed")

    assert runtime.state is None


def test_runtime_state_rejects_zero_mass_probability_rows() -> None:
    model = _make_model()
    runtime = HSMMFilterRuntime(model)
    runtime.step(torch.zeros(1, model.config.n_features))
    state = runtime.state
    assert state is not None

    bad_duration = state.duration_log_prob.clone()
    bad_duration[:, 0] = float("-inf")
    try:
        HSMMRuntimeState(
            filter_state=state.filter_state,
            observations=state.observations,
            duration_log_prob=bad_duration,
            transition_log_prob=state.transition_log_prob,
            encoder_state=state.encoder_state,
            last_timestamp=state.last_timestamp,
            uses_timestamps=state.uses_timestamps,
        )
    except ValueError as exc:
        assert "finite probability mass" in str(exc)
    else:
        raise AssertionError("zero-mass runtime probability rows must fail closed")
