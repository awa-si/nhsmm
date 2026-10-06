from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nhsmm import HSMMFilterRuntime, ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from nhsmm.filtering import (
    filter_model_sequence,
    filter_step,
    initialize_filter,
    next_episode_end_probability,
)
from tests.reference.hsmm_exact import (
    age_posterior_from_joint,
    causal_prefix_episode_end_probability,
    causal_prefix_joint_posterior,
    causal_prefix_log_evidence,
    state_posterior_from_joint,
)


def _scores(K: int, D: int, T: int, seed: int, mode: str):
    dtype = torch.float64
    g = torch.Generator().manual_seed(seed)
    initial = torch.randn(K, generator=g, dtype=dtype)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.3
    if mode == "uniform":
        initial.zero_()
        duration.zero_()
        transition.zero_()
    elif mode == "near_deterministic":
        initial *= 8.0
        duration *= 8.0
        transition *= 8.0
    elif mode == "impossible_support":
        if D > 1:
            duration[..., 0] = float("-inf")
        if K > 1:
            transition[..., 0] = float("-inf")
            transition[..., 1] = 0.0
    return (
        initial.log_softmax(-1),
        duration.log_softmax(-1),
        transition.log_softmax(-1),
        emission,
    )


def _sequence(emission: torch.Tensor) -> SequenceSet:
    T, _ = emission.shape
    return SequenceSet(
        sequences=torch.zeros(1, T, 1, dtype=emission.dtype),
        lengths=torch.tensor([T]),
        masks=torch.ones(1, T, 1, dtype=torch.bool),
        contexts=torch.zeros(1, T, 0, dtype=emission.dtype),
        canonical=torch.zeros(1, 1, 0, dtype=emission.dtype),
        log_probs=emission.unsqueeze(0),
    )


@pytest.mark.parametrize(
    "K,D,T", [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 4), (2, 3, 5), (3, 2, 4)]
)
@pytest.mark.parametrize("mode", ["uniform", "random", "near_deterministic", "impossible_support"])
def test_causal_forward_and_filter_model_sequence_match_exhaustive_posteriors(K, D, T, mode):
    initial, duration, transition, emission = _scores(K, D, T, 3000 + K * 100 + D * 10 + T, mode)
    sequence = _sequence(emission)
    expected_joint = causal_prefix_joint_posterior(initial, duration, transition, emission)
    expected_state = state_posterior_from_joint(expected_joint)
    expected_age = age_posterior_from_joint(expected_joint)
    expected_evidence = causal_prefix_log_evidence(initial, duration, transition, emission)
    expected_end = causal_prefix_episode_end_probability(initial, duration, transition, emission)

    model = NHSMM(
        ModelConfig(
            n_states=K, n_features=1, max_duration=D, causal=True, dropout=0.0, verbose=False
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with (
        patch.object(model, "_build_sequence_set", return_value=sequence),
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        alpha = model.forward(sequence)
        trace = filter_model_sequence(model, torch.zeros(1, T, 1, dtype=torch.float64))

    evidence = torch.logsumexp(alpha[0].flatten(1), dim=-1)
    normalized = alpha[0] - evidence[:, None, None]
    finite = torch.isfinite(expected_joint)
    torch.testing.assert_close(normalized[finite], expected_joint[finite], atol=3e-12, rtol=3e-12)
    assert torch.equal(torch.isneginf(normalized), torch.isneginf(expected_joint))
    torch.testing.assert_close(trace.log_posterior[0], expected_joint, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(trace.state_posterior[0], expected_state, atol=3e-12, rtol=3e-12)
    actual_age = torch.logsumexp(trace.log_posterior[0], dim=1).exp()
    torch.testing.assert_close(actual_age, expected_age, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(evidence, expected_evidence, atol=3e-12, rtol=3e-12)

    # Compare end probability directly from each exact filter slice through the public API.
    ends = []
    from nhsmm.filtering import HSMMFilterState

    for t in range(T):
        state = HSMMFilterState(expected_joint[t].unsqueeze(0))
        ends.append(next_episode_end_probability(state, duration[t]).squeeze(0))
    torch.testing.assert_close(torch.stack(ends), expected_end, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("K,D,T", [(1, 2, 4), (2, 2, 4), (2, 3, 5)])
def test_public_filter_step_matches_exhaustive_prefix_posterior(K, D, T):
    initial, duration, transition, emission = _scores(
        K, D, T, 4100 + K * 100 + D * 10 + T, "random"
    )
    expected = causal_prefix_joint_posterior(initial, duration, transition, emission)
    state = initialize_filter(initial, emission[0], D)
    torch.testing.assert_close(state.log_posterior[0], expected[0], atol=3e-12, rtol=3e-12)
    for t in range(1, T):
        state = filter_step(state, emission[t], duration[t - 1], transition[t - 1])
        torch.testing.assert_close(state.log_posterior[0], expected[t], atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("K,D,T", [(1, 2, 4), (2, 3, 5)])
@pytest.mark.parametrize("mode", ["uniform", "random", "near_deterministic", "impossible_support"])
def test_runtime_step_recursion_matches_exhaustive_oracle(monkeypatch, K, D, T, mode) -> None:
    initial, duration, transition, emission = _scores(K, D, T, 5151 + K * 100 + D * 10 + T, mode)
    expected = causal_prefix_joint_posterior(initial, duration, transition, emission)

    model = NHSMM(
        ModelConfig(
            n_states=K, n_features=1, max_duration=D, causal=True, dropout=0.0, verbose=False
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    model = model.to(dtype=torch.float64)

    step = {"i": 0}
    monkeypatch.setattr(model.dist.initial, "log_matrix", lambda **_: initial.view(1, 1, K))

    import nhsmm.runtime as runtime_module

    def fake_emission(_model, observation, context, cache):
        return emission[step["i"]].unsqueeze(0)

    def fake_boundary(_model, context, *, temperature, cache, batch_size=None):
        i = step["i"]
        return duration[i].unsqueeze(0), transition[i].unsqueeze(0)

    monkeypatch.setattr(runtime_module, "_emission_log_prob", fake_emission)
    monkeypatch.setattr(runtime_module, "_boundary_scores", fake_boundary)

    runtime = HSMMFilterRuntime(model)
    for t in range(T):
        step["i"] = t
        state = runtime.step(torch.zeros(1, 1, dtype=torch.float64), timestamp=t)
        torch.testing.assert_close(state.log_posterior[0], expected[t], atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("K,D,T", [(1, 2, 4), (2, 2, 4), (2, 3, 5), (3, 2, 4)])
@pytest.mark.parametrize("mode", ["uniform", "random", "near_deterministic", "impossible_support"])
def test_transition_forecast_matches_exhaustive_posterior(K, D, T, mode):
    from nhsmm.filtering import HSMMFilterState
    from nhsmm.transitions import one_step_transition_forecast
    from tests.reference.hsmm_exact import causal_one_step_transition_forecast

    initial, duration, transition, emission = _scores(K, D, T, 6200 + K * 100 + D * 10 + T, mode)
    expected_joint = causal_prefix_joint_posterior(initial, duration, transition, emission)

    for t in range(T):
        expected = causal_one_step_transition_forecast(
            expected_joint[t], duration[t], transition[t]
        )
        state = HSMMFilterState(expected_joint[t].unsqueeze(0))
        actual = one_step_transition_forecast(
            state,
            duration[t].unsqueeze(0),
            transition[t].unsqueeze(0),
        )
        torch.testing.assert_close(
            actual.episode_end_probability[0],
            expected["episode_end_probability"],
            atol=4e-12,
            rtol=4e-12,
        )
        torch.testing.assert_close(
            actual.boundary_transition_joint[0],
            expected["boundary_transition_joint"],
            atol=4e-12,
            rtol=4e-12,
        )
        torch.testing.assert_close(
            actual.next_episode_state_joint[0],
            expected["next_episode_state_joint"],
            atol=4e-12,
            rtol=4e-12,
        )
        torch.testing.assert_close(
            actual.next_state_prior[0],
            expected["next_state_prior"],
            atol=4e-12,
            rtol=4e-12,
        )
        torch.testing.assert_close(
            actual.state_change_probability[0],
            expected["state_change_probability"],
            atol=4e-12,
            rtol=4e-12,
        )
