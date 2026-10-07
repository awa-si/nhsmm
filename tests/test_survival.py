from __future__ import annotations

import torch

from nhsmm.filtering import HSMMFilterState, next_episode_end_probability
from nhsmm.survival import active_episode_survival_forecast


def _normalized_state(raw: torch.Tensor) -> HSMMFilterState:
    normalized = raw - torch.logsumexp(raw.flatten(1), dim=1).view(-1, 1, 1)
    return HSMMFilterState(normalized)


def test_deterministic_duration_survival_contract() -> None:
    log_posterior = torch.full((1, 1, 5), float("-inf"))
    log_posterior[..., 0] = 0.0  # current age = 1
    state = HSMMFilterState(log_posterior)

    log_duration = torch.full((1, 1, 5), float("-inf"))
    log_duration[..., 3] = 0.0  # total duration is exactly 4

    forecast = active_episode_survival_forecast(
        state,
        log_duration,
        horizons=(1, 2, 3, 4, 5),
    )

    assert torch.equal(
        forecast.survival_probability,
        torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        forecast.end_within_probability,
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]),
    )


def test_horizon_one_matches_existing_one_step_end_probability() -> None:
    torch.manual_seed(41)
    state = _normalized_state(torch.randn(2, 3, 5))
    log_duration = torch.randn(2, 3, 5)

    forecast = active_episode_survival_forecast(
        state,
        log_duration,
        horizons=(1, 2, 4),
    )
    one_step = next_episode_end_probability(state, log_duration)

    assert torch.allclose(
        forecast.end_within_probability[:, 0],
        one_step,
        atol=1e-6,
        rtol=1e-6,
    )


def test_survival_is_monotone_across_horizons() -> None:
    torch.manual_seed(43)
    state = _normalized_state(torch.randn(3, 2, 6))
    log_duration = torch.randn(3, 2, 6)

    forecast = active_episode_survival_forecast(
        state,
        log_duration,
        horizons=(1, 2, 3, 5, 7),
    )

    assert torch.all(
        forecast.survival_probability[:, 1:] <= forecast.survival_probability[:, :-1] + 1e-6
    )
    assert torch.all(
        forecast.end_within_probability[:, 1:] + 1e-6 >= forecast.end_within_probability[:, :-1]
    )
    assert torch.allclose(
        forecast.survival_probability + forecast.end_within_probability,
        torch.ones_like(forecast.survival_probability),
        atol=1e-6,
        rtol=1e-6,
    )


def test_horizons_must_be_positive_and_strictly_increasing() -> None:
    state = _normalized_state(torch.randn(1, 2, 4))
    log_duration = torch.randn(1, 2, 4)

    for horizons in ((0, 1), (2, 1), (1, 1)):
        try:
            active_episode_survival_forecast(state, log_duration, horizons)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid horizons must fail: {horizons}")


def test_tail_hazard_one_matches_finite_survival_exactly() -> None:
    torch.manual_seed(101)
    state = _normalized_state(torch.randn(2, 3, 5))
    log_duration = torch.randn(2, 3, 5)
    horizons = (1, 2, 4, 7)

    finite = active_episode_survival_forecast(state, log_duration, horizons)
    tailed = active_episode_survival_forecast(
        state,
        log_duration,
        horizons,
        torch.ones(2, 3),
    )

    torch.testing.assert_close(
        tailed.survival_probability, finite.survival_probability, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        tailed.end_within_probability, finite.end_within_probability, atol=0.0, rtol=0.0
    )


def test_tail_survival_extends_geometrically_beyond_max_duration() -> None:
    log_posterior = torch.full((1, 1, 3), float("-inf"))
    log_posterior[..., -1] = 0.0
    state = HSMMFilterState(log_posterior)
    log_duration = torch.log(torch.tensor([[[0.2, 0.3, 0.5]]]))
    tail = torch.tensor([[0.25]])

    forecast = active_episode_survival_forecast(
        state,
        log_duration,
        horizons=(1, 2, 5, 10),
        tail_end_probability=tail,
    )

    expected = torch.tensor(
        [[0.75, 0.75**2, 0.75**5, 0.75**10]], dtype=forecast.survival_probability.dtype
    )
    torch.testing.assert_close(forecast.survival_probability, expected, atol=1e-6, rtol=1e-6)


def test_tail_horizon_one_matches_tail_end_probability() -> None:
    torch.manual_seed(103)
    state = _normalized_state(torch.randn(2, 3, 4))
    log_duration = torch.randn(2, 3, 4)
    tail = torch.full((2, 3), 0.35)

    forecast = active_episode_survival_forecast(
        state, log_duration, horizons=(1, 3), tail_end_probability=tail
    )
    one_step = next_episode_end_probability(state, log_duration, tail)

    torch.testing.assert_close(
        forecast.end_within_probability[:, 0], one_step, atol=1e-6, rtol=1e-6
    )


def test_tail_survival_supports_large_horizon_without_unbounded_age_state() -> None:
    log_posterior = torch.tensor([[[float("-inf"), 0.0]]])
    state = HSMMFilterState(log_posterior)
    log_duration = torch.log(torch.tensor([[[0.4, 0.6]]]))
    tail = torch.tensor([[0.1]])

    forecast = active_episode_survival_forecast(
        state,
        log_duration,
        horizons=(10_000,),
        tail_end_probability=tail,
    )

    expected = torch.tensor([[0.9**10_000]], dtype=forecast.survival_probability.dtype)
    torch.testing.assert_close(forecast.survival_probability, expected, atol=1e-7, rtol=1e-6)
