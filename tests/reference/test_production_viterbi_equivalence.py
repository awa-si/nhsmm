from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from tests.reference.hsmm_exact import (
    causal_map_state_age_path,
    enumerate_causal_paths,
    enumerate_segment_paths,
    segment_map_state_path,
)


def _scores(K: int, D: int, T: int, seed: int, mode: str):
    dtype = torch.float64
    g = torch.Generator().manual_seed(seed)
    initial = torch.randn(K, generator=g, dtype=dtype)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.2
    if mode == "uniform":
        initial.zero_()
        duration.zero_()
        transition.zero_()
        emission.zero_()
    elif mode == "near_deterministic":
        initial *= 10.0
        duration *= 10.0
        transition *= 10.0
        emission *= 4.0
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


def _compatible_causal_score(paths, state_path: list[int]) -> torch.Tensor:
    compatible = [p.log_prob for p in paths if list(p.states) == state_path]
    if not compatible:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return torch.stack(compatible).max()


def _segment_state_path(path) -> list[int]:
    states: list[int] = []
    for state, duration in zip(path.states, path.durations):
        states.extend([state] * duration)
    return states


def _compatible_segment_score(paths, state_path: list[int]) -> torch.Tensor:
    compatible = [p.log_prob for p in paths if _segment_state_path(p) == state_path]
    if not compatible:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return torch.stack(compatible).max()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "K,D,T",
    [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 4), (2, 3, 5), (3, 2, 4)],
)
@pytest.mark.parametrize("mode", ["uniform", "random", "near_deterministic", "impossible_support"])
def test_production_viterbi_path_has_global_exhaustive_map_score(causal, K, D, T, mode):
    initial, duration, transition, emission = _scores(
        K, D, T, 7100 + K * 100 + D * 10 + T + int(causal), mode
    )
    sequence = _sequence(emission)
    model = NHSMM(
        ModelConfig(
            n_states=K,
            n_features=1,
            max_duration=D,
            causal=causal,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        actual_path = model._viterbi(sequence)[0].tolist()

    if causal:
        paths = enumerate_causal_paths(initial, duration, transition, emission)
        _, _, expected_score = causal_map_state_age_path(paths)
        actual_score = _compatible_causal_score(paths, actual_path)
    else:
        paths = enumerate_segment_paths(initial, duration, transition, emission)
        _, expected_score = segment_map_state_path(paths)
        actual_score = _compatible_segment_score(paths, actual_path)

    torch.testing.assert_close(actual_score, expected_score, atol=3e-12, rtol=3e-12)


def test_noncausal_viterbi_global_choice_can_beat_local_predecessor_duration() -> None:
    K, D, T = 2, 2, 3
    initial = torch.log(torch.tensor([1.0, 1e-30], dtype=torch.float64))
    duration = torch.log(
        torch.tensor(
            [
                [[1.0, 1e-30], [1.0, 1e-30]],
                [[0.525, 0.475], [1.0, 1e-30]],
                [[1.0, 1e-30], [1.0, 1e-30]],
            ],
            dtype=torch.float64,
        )
    )
    transition = torch.full((T, K, D, K), -100.0, dtype=torch.float64)
    transition[0, 0, 0, 0] = 0.0
    transition[1, 0, 0] = torch.tensor([0.0, -10.0], dtype=torch.float64)
    transition[1, 0, 1] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    transition = transition.log_softmax(-1)
    emission = torch.zeros(T, K, dtype=torch.float64)
    emission[2, 0] = -5.0

    paths = enumerate_segment_paths(initial, duration, transition, emission)
    expected_path, expected_score = segment_map_state_path(paths)
    assert expected_path == [0, 0, 1]

    sequence = _sequence(emission)
    model = NHSMM(
        ModelConfig(n_states=K, n_features=1, max_duration=D, causal=False, dropout=0.0),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        actual = model._viterbi(sequence)[0].tolist()
    assert actual == expected_path
    torch.testing.assert_close(_compatible_segment_score(paths, actual), expected_score)


def test_causal_viterbi_self_transition_reset_affects_future_map_choice() -> None:
    K, D, T = 2, 2, 4
    initial = torch.tensor([0.0, float("-inf")], dtype=torch.float64)
    # Every episode has duration one, so a self-transition at t=1 is a new episode
    # and must reset age to one. The following boundary then prefers state 1.
    duration = torch.full((T, K, D), float("-inf"), dtype=torch.float64)
    duration[..., 0] = 0.0
    transition = torch.full((T, K, D, K), float("-inf"), dtype=torch.float64)
    transition[:, 0, 0, 0] = 0.0
    transition[:, 0, 1, 0] = 0.0  # unreachable age-2 row remains normalized
    transition[1, 0, 0, 1] = 1.0
    transition[1, 0, 0, 0] = 0.0
    transition[:, 1, 0, 1] = 0.0
    transition[:, 1, 1, 1] = 0.0  # unreachable age-2 row remains normalized
    transition = transition.log_softmax(-1)
    emission = torch.zeros(T, K, dtype=torch.float64)
    emission[0, 1] = -20.0
    emission[1, 1] = -20.0
    emission[2:, 0] = -5.0

    paths = enumerate_causal_paths(initial, duration, transition, emission)
    expected_states, expected_ages, expected_score = causal_map_state_age_path(paths)
    assert expected_states == [0, 0, 1, 1]
    assert expected_ages == [1, 1, 1, 1]

    sequence = _sequence(emission)
    model = NHSMM(
        ModelConfig(n_states=K, n_features=1, max_duration=D, causal=True, dropout=0.0),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        actual = model._viterbi(sequence)[0].tolist()

    assert actual == expected_states
    torch.testing.assert_close(_compatible_causal_score(paths, actual), expected_score)


def test_noncausal_viterbi_preserves_float64_close_margin_map_choice() -> None:
    K, D, T = 2, 2, 4
    scale = 2e-7
    g = torch.Generator().manual_seed(1)
    initial = (torch.randn(K, generator=g, dtype=torch.float64) * scale).log_softmax(-1)
    duration = (torch.randn(T, K, D, generator=g, dtype=torch.float64) * scale).log_softmax(-1)
    transition = (torch.randn(T, K, D, K, generator=g, dtype=torch.float64) * scale).log_softmax(-1)
    emission = torch.randn(T, K, generator=g, dtype=torch.float64) * scale

    paths = enumerate_segment_paths(initial, duration, transition, emission)
    expected_path, expected_score = segment_map_state_path(paths)

    sequence = _sequence(emission)
    model = NHSMM(
        ModelConfig(
            n_states=K,
            n_features=1,
            max_duration=D,
            causal=False,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        actual = model._viterbi(sequence)[0].tolist()

    assert actual == expected_path
    torch.testing.assert_close(
        _compatible_segment_score(paths, actual),
        expected_score,
        atol=1e-12,
        rtol=1e-12,
    )
