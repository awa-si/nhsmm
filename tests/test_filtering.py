from __future__ import annotations

import math

import pytest
import torch

from nhsmm.filtering import (
    HSMMFilterState,
    HSMMFilterTrace,
    _filter_step_normalized,
    duration_log_hazard,
    filter_step,
    initialize_filter,
    next_episode_end_probability,
)


def _deterministic_duration(K: int, D: int, duration: int) -> torch.Tensor:
    logits = torch.full((1, K, D), float("-inf"))
    logits[..., duration - 1] = 0.0
    return logits


def _identity_transition(K: int, D: int) -> torch.Tensor:
    logits = torch.full((1, K, D, K), float("-inf"))
    for k in range(K):
        logits[:, k, :, k] = 0.0
    return logits


def test_duration_hazard_deterministic() -> None:
    duration = _deterministic_duration(K=1, D=4, duration=3)
    log_end, log_continue = duration_log_hazard(duration)

    assert torch.allclose(log_end.exp()[0, 0, :3], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.isneginf(log_end[0, 0, 3])
    assert torch.allclose(log_continue.exp()[0, 0, :2], torch.tensor([1.0, 1.0]))
    assert torch.isneginf(log_continue[0, 0, 2:]).all()


def test_duration_rejects_zero_mass_rows() -> None:
    with pytest.raises(ValueError, match="log_duration has zero or non-finite probability mass"):
        duration_log_hazard(torch.full((1, 2, 3), float("-inf")))


def test_filter_age_progression_and_reset() -> None:
    D = 3
    duration = _deterministic_duration(K=1, D=D, duration=3)
    transition = _identity_transition(K=1, D=D)
    emission = torch.zeros(1, 1)

    state = initialize_filter(torch.zeros(1, 1), emission, max_duration=D)
    assert torch.allclose(state.age_posterior, torch.tensor([[1.0, 0.0, 0.0]]))

    state = filter_step(state, emission, duration, transition)
    assert torch.allclose(state.age_posterior, torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(next_episode_end_probability(state, duration), torch.tensor([0.0]))

    state = filter_step(state, emission, duration, transition)
    assert torch.allclose(state.age_posterior, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(next_episode_end_probability(state, duration), torch.tensor([1.0]))

    state = filter_step(state, emission, duration, transition)
    assert torch.allclose(state.age_posterior, torch.tensor([[1.0, 0.0, 0.0]]))


def test_transition_occurs_only_at_episode_boundary() -> None:
    K, D = 2, 2
    duration = _deterministic_duration(K=K, D=D, duration=2)
    transition = torch.full((1, K, D, K), float("-inf"))
    transition[:, 0, :, 1] = 0.0
    transition[:, 1, :, 1] = 0.0
    emission = torch.zeros(1, K)

    state = initialize_filter(torch.tensor([[0.0, -math.inf]]), emission, max_duration=D)
    assert torch.allclose(state.state_posterior, torch.tensor([[1.0, 0.0]]))

    state = filter_step(state, emission, duration, transition)
    assert torch.allclose(state.state_posterior, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(state.age_posterior, torch.tensor([[0.0, 1.0]]))

    state = filter_step(state, emission, duration, transition)
    assert torch.allclose(state.state_posterior, torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(state.age_posterior, torch.tensor([[1.0, 0.0]]))


def test_filter_rejects_zero_mass_transition_rows() -> None:
    state = initialize_filter(torch.zeros(1, 1), torch.zeros(1, 1), max_duration=2)
    with pytest.raises(
        ValueError, match="transition_log_prob has zero or non-finite probability mass"
    ):
        filter_step(
            state,
            torch.zeros(1, 1),
            _deterministic_duration(K=1, D=2, duration=2),
            torch.full((1, 1, 2, 1), float("-inf")),
        )


def test_filter_rejects_dtype_mismatch() -> None:
    state = initialize_filter(torch.zeros(1, 1), torch.zeros(1, 1), max_duration=2)
    with pytest.raises(ValueError, match="dtype"):
        filter_step(
            state,
            torch.zeros(1, 1, dtype=torch.float64),
            _deterministic_duration(K=1, D=2, duration=2),
            _identity_transition(K=1, D=2),
        )


def test_filter_state_rejects_nan() -> None:
    bad = torch.zeros(1, 1, 2)
    bad[..., 1] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        HSMMFilterState(bad)


def test_internal_normalized_filter_matches_public_filter() -> None:
    torch.manual_seed(17)
    B, K, D = 2, 3, 4
    state = initialize_filter(torch.randn(B, K), torch.randn(B, K), max_duration=D)
    emission = torch.randn(B, K)

    duration = torch.randn(B, K, D)
    duration[:, :, 0] = float("-inf")
    duration = torch.log_softmax(duration, dim=-1)
    transition = torch.log_softmax(torch.randn(B, K, D, K), dim=-1)

    reference = filter_step(state, emission, duration, transition)
    fast = _filter_step_normalized(state, emission, duration, transition)

    assert torch.allclose(
        fast.log_posterior,
        reference.log_posterior,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        fast.log_posterior.flatten(1).logsumexp(dim=1),
        torch.zeros(B),
        atol=1e-6,
        rtol=1e-6,
    )


def test_filter_posterior_normalizes() -> None:
    torch.manual_seed(11)
    B, K, D = 3, 4, 5
    state = initialize_filter(torch.randn(B, K), torch.randn(B, K), max_duration=D)
    for _ in range(8):
        state = filter_step(
            state,
            torch.randn(B, K),
            torch.randn(B, K, D),
            torch.randn(B, K, D, K),
        )
        assert torch.allclose(
            state.log_posterior.flatten(1).logsumexp(dim=1),
            torch.zeros(B),
            atol=1e-6,
            rtol=1e-6,
        )


def test_filter_trace_requires_long_lengths_on_same_device() -> None:
    trace = torch.zeros(1, 2, 1, 1)
    with pytest.raises(TypeError, match="torch.long"):
        HSMMFilterTrace(trace, torch.tensor([2.0]))

    meta_lengths = torch.tensor([2], dtype=torch.long, device="meta")
    with pytest.raises(ValueError, match="same device"):
        HSMMFilterTrace(trace, meta_lengths)


def test_tail_hazard_one_is_exactly_equivalent_to_finite_filter_step() -> None:
    torch.manual_seed(101)
    B, K, D = 2, 3, 4
    state = initialize_filter(torch.randn(B, K), torch.randn(B, K), max_duration=D)
    emission = torch.randn(B, K)
    duration = torch.log_softmax(torch.randn(B, K, D), dim=-1)
    transition = torch.log_softmax(torch.randn(B, K, D, K), dim=-1)

    finite = _filter_step_normalized(state, emission, duration, transition)
    tailed = _filter_step_normalized(
        state,
        emission,
        duration,
        transition,
        torch.ones(B, K),
    )

    torch.testing.assert_close(tailed.log_posterior, finite.log_posterior, atol=0.0, rtol=0.0)


def test_tail_bucket_accepts_entry_and_self_continuation() -> None:
    D = 2
    duration_mass = torch.tensor([[[float("-inf"), 0.0]]])
    transition = _identity_transition(K=1, D=D)
    emission = torch.zeros(1, 1)
    tail = torch.tensor([[0.25]])

    state = initialize_filter(torch.zeros(1, 1), emission, max_duration=D)
    state = filter_step(state, emission, duration_mass, transition, tail)
    torch.testing.assert_close(state.age_posterior, torch.tensor([[0.0, 1.0]]))

    state = filter_step(state, emission, duration_mass, transition, tail)
    torch.testing.assert_close(
        state.age_posterior,
        torch.tensor([[0.25, 0.75]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_d1_tail_combines_boundary_reset_and_tail_continuation() -> None:
    state = initialize_filter(torch.zeros(1, 1), torch.zeros(1, 1), max_duration=1)
    duration_mass = torch.zeros(1, 1, 1)
    transition = torch.zeros(1, 1, 1, 1)

    for _ in range(4):
        state = filter_step(
            state,
            torch.zeros(1, 1),
            duration_mass,
            transition,
            torch.tensor([[0.2]]),
        )
        torch.testing.assert_close(state.age_posterior, torch.ones(1, 1))


def test_safe_logaddexp_has_finite_gradient_for_two_impossible_paths() -> None:
    from nhsmm.filtering import _safe_logaddexp

    left = torch.tensor([float("-inf"), -2.0], requires_grad=True)
    right = torch.tensor([float("-inf"), -3.0], requires_grad=True)
    result = _safe_logaddexp(left, right)

    assert torch.isneginf(result[0])
    result[1].backward()
    assert left.grad is not None
    assert right.grad is not None
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()
