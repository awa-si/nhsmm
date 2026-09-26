from __future__ import annotations

import torch

from nhsmm.filtering import HSMMFilterState, next_episode_end_probability
from nhsmm.transitions import one_step_transition_forecast


def _state(state: int, age: int, *, K: int = 2, D: int = 2) -> HSMMFilterState:
    log_posterior = torch.full((1, K, D), float("-inf"), dtype=torch.float64)
    log_posterior[0, state, age - 1] = 0.0
    return HSMMFilterState(log_posterior)


def _duration_d2() -> torch.Tensor:
    # Both states have deterministic total duration 2.
    return torch.tensor(
        [[[-float("inf"), 0.0], [-float("inf"), 0.0]]],
        dtype=torch.float64,
    )


def test_continuing_episode_keeps_state_prior() -> None:
    state = _state(0, 1)
    transition = torch.zeros((1, 2, 2, 2), dtype=torch.float64)

    forecast = one_step_transition_forecast(state, _duration_d2(), transition)

    assert torch.allclose(forecast.episode_end_probability, torch.tensor([0.0], dtype=torch.float64))
    assert torch.allclose(forecast.next_state_prior, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert torch.count_nonzero(forecast.boundary_transition_joint) == 0
    assert torch.allclose(forecast.state_change_probability, torch.tensor([0.0], dtype=torch.float64))


def test_boundary_transition_and_state_change_are_distinct() -> None:
    state = _state(0, 2)
    transition = torch.full((1, 2, 2, 2), float("-inf"), dtype=torch.float64)
    transition[0, 0, :, 1] = 0.0
    transition[0, 1, :, 1] = 0.0

    forecast = one_step_transition_forecast(state, _duration_d2(), transition)

    assert torch.allclose(forecast.episode_end_probability, torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(forecast.boundary_transition_joint[0, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(forecast.next_episode_state_joint, torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    assert torch.allclose(forecast.next_state_prior, torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    assert torch.allclose(forecast.state_change_probability, torch.tensor([1.0], dtype=torch.float64))


def test_self_transition_is_boundary_but_not_state_change() -> None:
    state = _state(0, 2)
    transition = torch.full((1, 2, 2, 2), float("-inf"), dtype=torch.float64)
    transition[0, 0, :, 0] = 0.0
    transition[0, 1, :, 1] = 0.0

    forecast = one_step_transition_forecast(state, _duration_d2(), transition)

    assert torch.allclose(forecast.episode_end_probability, torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(forecast.next_state_prior, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(forecast.state_change_probability, torch.tensor([0.0], dtype=torch.float64))


def test_transition_end_probability_matches_existing_hazard() -> None:
    torch.manual_seed(41)
    posterior = torch.log_softmax(torch.randn(2, 3, 4, dtype=torch.float64).flatten(1), dim=1).reshape(2, 3, 4)
    state = HSMMFilterState(posterior)
    duration = torch.randn(2, 3, 4, dtype=torch.float64)
    transition = torch.randn(2, 3, 4, 3, dtype=torch.float64)

    forecast = one_step_transition_forecast(state, duration, transition)
    existing = next_episode_end_probability(state, duration)

    assert torch.allclose(forecast.episode_end_probability, existing, atol=1e-12, rtol=1e-12)
    assert torch.allclose(forecast.next_state_prior.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert torch.allclose(
        forecast.boundary_transition_joint.sum(dim=(1, 2)),
        forecast.episode_end_probability,
        atol=1e-12,
        rtol=0.0,
    )
