from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from tests.reference.hsmm_exact import (
    causal_log_likelihood,
    enumerate_causal_paths,
    enumerate_segment_paths,
    segment_log_likelihood,
)


def _scores(K: int, D: int, T: int, seed: int, mode: str):
    dtype = torch.float64
    g = torch.Generator().manual_seed(seed)
    initial = torch.randn(K, generator=g, dtype=dtype)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.4
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


def _sequence(emission: torch.Tensor, length: int | None = None) -> SequenceSet:
    T, _ = emission.shape
    L = T if length is None else length
    return SequenceSet(
        sequences=torch.zeros(1, T, 1, dtype=emission.dtype),
        lengths=torch.tensor([L]),
        masks=(torch.arange(T) < L).view(1, T, 1),
        contexts=torch.zeros(1, T, 0, dtype=emission.dtype),
        canonical=torch.zeros(1, 1, 0, dtype=emission.dtype),
        log_probs=emission.unsqueeze(0),
    )


def _expected(causal, initial, duration, transition, emission):
    if causal:
        return causal_log_likelihood(
            enumerate_causal_paths(initial, duration, transition, emission)
        )
    return segment_log_likelihood(enumerate_segment_paths(initial, duration, transition, emission))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "K,D,T", [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 3), (2, 3, 5), (3, 2, 4)]
)
@pytest.mark.parametrize("mode", ["uniform", "random", "near_deterministic", "impossible_support"])
def test_production_forward_likelihood_matches_exhaustive_reference(causal, K, D, T, mode):
    initial, duration, transition, emission = _scores(K, D, T, 1000 + K * 100 + D * 10 + T, mode)
    sequence = _sequence(emission)
    model = NHSMM(
        ModelConfig(
            n_states=K, n_features=1, max_duration=D, causal=causal, dropout=0.0, verbose=False
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
        alpha = model.forward(sequence)
        actual = torch.logsumexp(alpha[0, T - 1].flatten(), dim=0)
        with patch.object(model, "_build_sequence_set", return_value=sequence):
            public = model.log_likelihood(torch.zeros(1, T, 1), reduce=False)[0]
    expected = _expected(causal, initial, duration, transition, emission)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(public, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("causal", [False, True])
def test_variable_length_batch_matches_per_sequence_reference(causal):
    K, D, T = 2, 3, 5
    lengths = [5, 3, 1]
    items = [_scores(K, D, T, 2000 + i, "random") for i in range(3)]
    initial = torch.stack([x[0] for x in items])
    duration = torch.stack([x[1] for x in items])
    transition = torch.stack([x[2] for x in items])
    emission = torch.stack([x[3] for x in items])
    mask = torch.arange(T).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    sequence = SequenceSet(
        sequences=torch.zeros(3, T, 1, dtype=torch.float64),
        lengths=torch.tensor(lengths),
        masks=mask.unsqueeze(-1),
        contexts=torch.zeros(3, T, 0, dtype=torch.float64),
        canonical=torch.zeros(3, 1, 0, dtype=torch.float64),
        log_probs=emission,
    )
    model = NHSMM(
        ModelConfig(
            n_states=K, n_features=1, max_duration=D, causal=causal, dropout=0.0, verbose=False
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.unsqueeze(1)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration),
        patch.object(model.dist.transition, "log_matrix", return_value=transition),
    ):
        alpha = model.forward(sequence)
    actual = torch.stack(
        [torch.logsumexp(alpha[b, L - 1].flatten(), 0) for b, L in enumerate(lengths)]
    )
    expected = []
    for b, L in enumerate(lengths):
        expected.append(
            _expected(causal, initial[b], duration[b, :L], transition[b, :L], emission[b, :L])
        )
    torch.testing.assert_close(actual, torch.stack(expected), atol=2e-12, rtol=2e-12)
