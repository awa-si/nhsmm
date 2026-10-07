from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from tests.reference.hsmm_exact import (
    enumerate_segment_paths,
    enumerate_segment_paths_with_tail,
    segment_log_likelihood,
    segment_map_state_path,
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


def _scores(K: int, D: int, T: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    initial = torch.randn(K, generator=g, dtype=dtype).log_softmax(-1)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype).log_softmax(-1)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype).log_softmax(-1)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.3
    tail = torch.sigmoid(torch.randn(K, generator=g, dtype=dtype)).clamp(0.1, 0.9)
    return initial, duration, transition, emission, tail


def _model(K: int, D: int, tail: torch.Tensor) -> NHSMM:
    model = NHSMM(
        ModelConfig(
            n_states=K,
            n_features=1,
            max_duration=D,
            causal=False,
            duration_tail=True,
            duration_tail_init_probability=0.5,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with torch.no_grad():
        model.dist.duration.tail_end_probability.copy_(tail)
    return model


@pytest.mark.parametrize("K,D,T", [(1, 1, 5), (1, 2, 5), (2, 2, 5), (2, 3, 6)])
def test_noncausal_tail_likelihood_matches_exhaustive_reference(K: int, D: int, T: int) -> None:
    initial, duration, transition, emission, tail = _scores(K, D, T, 8100 + K * 100 + D * 10 + T)
    paths = enumerate_segment_paths_with_tail(initial, duration, transition, emission, tail)
    expected = segment_log_likelihood(paths)
    sequence = _sequence(emission)
    model = _model(K, D, tail)

    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
        patch.object(model.dist.duration, "tail_probability", return_value=tail),
    ):
        alpha = model.forward(sequence)
        actual = torch.logsumexp(alpha[0, T - 1].flatten(), dim=0)

    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("K,D,T", [(1, 1, 5), (2, 2, 5), (2, 3, 6)])
def test_noncausal_tail_viterbi_has_exhaustive_map_score(K: int, D: int, T: int) -> None:
    initial, duration, transition, emission, tail = _scores(K, D, T, 8200 + K * 100 + D * 10 + T)
    paths = enumerate_segment_paths_with_tail(initial, duration, transition, emission, tail)
    expected_path, expected_score = segment_map_state_path(paths)
    sequence = _sequence(emission)
    model = _model(K, D, tail)

    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
        patch.object(model.dist.duration, "tail_probability", return_value=tail),
    ):
        actual_path = model._viterbi(sequence)[0].tolist()

    compatible = []
    for path in paths:
        expanded = []
        for state, length in zip(path.states, path.durations):
            expanded.extend([state] * length)
        if expanded == actual_path:
            compatible.append(path.log_prob)
    actual_score = torch.stack(compatible).max()
    assert actual_path == expected_path or torch.allclose(actual_score, expected_score)
    torch.testing.assert_close(actual_score, expected_score, atol=3e-12, rtol=3e-12)


def test_tail_hazard_one_reduces_to_finite_noncausal_reference_and_production() -> None:
    K, D, T = 2, 3, 6
    initial, duration, transition, emission, _ = _scores(K, D, T, 8301)
    tail = torch.ones(K, dtype=torch.float64)
    finite_paths = enumerate_segment_paths(initial, duration, transition, emission)
    tail_paths = enumerate_segment_paths_with_tail(initial, duration, transition, emission, tail)
    finite_ll = segment_log_likelihood(finite_paths)
    tail_ll = segment_log_likelihood(tail_paths)
    torch.testing.assert_close(tail_ll, finite_ll, atol=0.0, rtol=0.0)

    finite = NHSMM(
        ModelConfig(n_states=K, n_features=1, max_duration=D, causal=False, dropout=0.0),
        device="cpu",
    )
    finite.initialize_distributions(jitter=0.0)
    finite.eval()
    tailed = _model(K, D, tail)
    sequence = _sequence(emission)
    with (
        patch.object(finite.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(finite.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(finite.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        finite_alpha = finite.forward(sequence)
        finite_path = finite._viterbi(sequence)[0]
    with (
        patch.object(tailed.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(tailed.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(tailed.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
        patch.object(tailed.dist.duration, "tail_probability", return_value=tail),
    ):
        tail_alpha = tailed.forward(sequence)
        tail_path = tailed._viterbi(sequence)[0]

    finite_ll_prod = torch.logsumexp(finite_alpha[0, T - 1].flatten(), dim=0)
    tail_ll_prod = torch.logsumexp(tail_alpha[0, T - 1].flatten(), dim=0)
    torch.testing.assert_close(tail_ll_prod, finite_ll_prod, atol=3e-12, rtol=3e-12)
    assert torch.equal(tail_path, finite_path)


def test_noncausal_tail_likelihood_backpropagates_to_tail_parameter() -> None:
    K, D, T = 1, 2, 5
    dtype = torch.float32
    initial = torch.zeros(K, dtype=dtype)
    duration = torch.log(torch.tensor([[[0.25, 0.75]]] * T, dtype=dtype))
    transition = torch.zeros(T, K, D, K, dtype=dtype)
    emission = torch.zeros(T, K, dtype=dtype)
    sequence = _sequence(emission)
    model = _model(K, D, torch.tensor([0.4], dtype=torch.float64))
    with torch.no_grad():
        model.dist.duration.tail_end_probability.fill_(0.4)
    model.dist.duration.tail_end_probability.grad = None

    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        alpha = model.forward(sequence)
        loss = -torch.logsumexp(alpha[0, T - 1].flatten(), dim=0)
        loss.backward()

    grad = model.dist.duration.tail_end_probability.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
