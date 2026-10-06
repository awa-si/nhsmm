from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from nhsmm.filtering import filter_step, initialize_filter
from tests.reference.hsmm_exact import (
    causal_log_likelihood,
    causal_prefix_joint_posterior,
    causal_prefix_log_evidence,
    enumerate_causal_paths,
    enumerate_segment_paths,
    segment_log_likelihood,
)

PROPERTY_SEEDS = tuple(range(24))


def _case(seed: int, *, dtype: torch.dtype = torch.float64):
    g = torch.Generator().manual_seed(100_000 + seed)
    K = 1 + (seed % 3)
    D = 1 + ((seed // 3) % 3)
    T = 1 + ((seed // 9) % 6)
    T = min(T + (seed % 3), 6)
    initial = torch.randn(K, generator=g, dtype=dtype)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.3
    scale = (0.5, 1.0, 5.0, 10.0)[seed % 4]
    initial *= scale
    duration *= scale
    transition *= scale
    return (
        K,
        D,
        T,
        initial.log_softmax(-1),
        duration.log_softmax(-1),
        transition.log_softmax(-1),
        emission,
    )


def _same_shape_case(K: int, D: int, T: int, seed: int, *, dtype: torch.dtype = torch.float64):
    g = torch.Generator().manual_seed(200_000 + seed)
    initial = torch.randn(K, generator=g, dtype=dtype).log_softmax(-1)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype).log_softmax(-1)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype).log_softmax(-1)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.3
    return initial, duration, transition, emission


def _sequence(emission: torch.Tensor, *, length: int | None = None) -> SequenceSet:
    T, _ = emission.shape
    L = T if length is None else length
    mask = (torch.arange(T) < L).view(1, T, 1)
    return SequenceSet(
        sequences=torch.zeros(1, T, 1, dtype=emission.dtype),
        lengths=torch.tensor([L]),
        masks=mask,
        contexts=torch.zeros(1, T, 0, dtype=emission.dtype),
        canonical=torch.zeros(1, 1, 0, dtype=emission.dtype),
        log_probs=emission.unsqueeze(0),
    )


def _model(K: int, D: int, *, causal: bool = True) -> NHSMM:
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
    return model


def _forward(model, sequence, initial, duration, transition):
    K = initial.shape[-1]
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        return model.forward(sequence)


@pytest.mark.parametrize("seed", PROPERTY_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_property_causal_posterior_normalization_finiteness_and_age_bound(seed: int) -> None:
    K, D, T, initial, duration, transition, emission = _case(seed)
    alpha = _forward(_model(K, D), _sequence(emission), initial, duration, transition)[0]
    evidence = torch.logsumexp(alpha.flatten(1), dim=-1)
    assert torch.isfinite(evidence).all(), f"seed={seed}"
    normalized = alpha - evidence[:, None, None]
    mass = normalized.exp().sum(dim=(1, 2))
    torch.testing.assert_close(mass, torch.ones(T, dtype=mass.dtype), atol=3e-12, rtol=3e-12)
    positive = (normalized.exp() > 0).nonzero()
    if positive.numel():
        assert int(positive[:, 2].max()) < D, f"seed={seed}"


@pytest.mark.parametrize("seed", PROPERTY_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_property_causal_prefix_and_right_padding_invariance(seed: int) -> None:
    K, D, T, initial, duration, transition, emission = _case(seed)
    if T == 1:
        return
    prefix = 1 + (seed % (T - 1))
    changed = emission.clone()
    changed[prefix:] += torch.linspace(-20.0, 20.0, K, dtype=emission.dtype)
    model = _model(K, D)
    baseline = _forward(model, _sequence(emission), initial, duration, transition)
    future_changed = _forward(model, _sequence(changed), initial, duration, transition)
    torch.testing.assert_close(baseline[:, :prefix], future_changed[:, :prefix], atol=0.0, rtol=0.0)

    padded = emission.clone()
    padded[prefix:] = torch.randn_like(padded[prefix:]) * 100.0
    masked_a = _forward(model, _sequence(emission, length=prefix), initial, duration, transition)
    masked_b = _forward(model, _sequence(padded, length=prefix), initial, duration, transition)
    torch.testing.assert_close(masked_a[:, :prefix], masked_b[:, :prefix], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("seed", PROPERTY_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_property_streaming_filter_equals_batch_prefixes(seed: int) -> None:
    K, D, T, initial, duration, transition, emission = _case(seed)
    alpha = _forward(_model(K, D), _sequence(emission), initial, duration, transition)[0]
    evidence = torch.logsumexp(alpha.flatten(1), dim=-1)
    expected = alpha - evidence[:, None, None]

    state = initialize_filter(initial, emission[0], D)
    torch.testing.assert_close(state.log_posterior[0], expected[0], atol=4e-12, rtol=4e-12)
    for t in range(1, T):
        state = filter_step(state, emission[t], duration[t - 1], transition[t - 1])
        torch.testing.assert_close(state.log_posterior[0], expected[t], atol=4e-12, rtol=4e-12)


@pytest.mark.parametrize("seed", PROPERTY_SEEDS[:12], ids=lambda seed: f"seed-{seed}")
def test_property_batch_matches_single_sequences(seed: int) -> None:
    K, D, T, initial_a, duration_a, transition_a, emission_a = _case(seed)
    initial_b, duration_b, transition_b, emission_b = _same_shape_case(K, D, T, seed + 24)
    length_b = max(1, T - 1)
    initial = torch.stack([initial_a, initial_b])
    duration = torch.stack([duration_a, duration_b])
    transition = torch.stack([transition_a, transition_b])
    emission = torch.stack([emission_a, emission_b])
    lengths = torch.tensor([T, length_b])
    mask = (torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
    sequence = SequenceSet(
        sequences=torch.zeros(2, T, 1, dtype=torch.float64),
        lengths=lengths,
        masks=mask,
        contexts=torch.zeros(2, T, 0, dtype=torch.float64),
        canonical=torch.zeros(2, 1, 0, dtype=torch.float64),
        log_probs=emission,
    )
    model = _model(K, D)
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.unsqueeze(1)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration),
        patch.object(model.dist.transition, "log_matrix", return_value=transition),
    ):
        batch = model.forward(sequence)
    for b, L in enumerate(lengths.tolist()):
        single = _forward(
            model,
            _sequence(emission[b], length=L),
            initial[b],
            duration[b],
            transition[b],
        )
        torch.testing.assert_close(batch[b, :L], single[0, :L], atol=4e-12, rtol=4e-12)


@pytest.mark.parametrize("seed", PROPERTY_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_property_production_likelihood_matches_reference_decomposition(seed: int) -> None:
    K, D, T, initial, duration, transition, emission = _case(seed)
    causal_paths = enumerate_causal_paths(initial, duration, transition, emission)
    prefix_evidence = causal_prefix_log_evidence(initial, duration, transition, emission)
    causal_expected = causal_log_likelihood(causal_paths)
    torch.testing.assert_close(prefix_evidence[-1], causal_expected, atol=3e-12, rtol=3e-12)

    causal_alpha = _forward(_model(K, D), _sequence(emission), initial, duration, transition)
    causal_actual = torch.logsumexp(causal_alpha[0, T - 1].flatten(), dim=0)
    torch.testing.assert_close(causal_actual, causal_expected, atol=4e-12, rtol=4e-12)

    segment_paths = enumerate_segment_paths(initial, duration, transition, emission)
    segment_expected = segment_log_likelihood(segment_paths)
    segment_alpha = _forward(
        _model(K, D, causal=False), _sequence(emission), initial, duration, transition
    )
    segment_actual = torch.logsumexp(segment_alpha[0, T - 1].flatten(), dim=0)
    torch.testing.assert_close(segment_actual, segment_expected, atol=4e-12, rtol=4e-12)


@pytest.mark.parametrize("seed", PROPERTY_SEEDS[:12], ids=lambda seed: f"seed-{seed}")
def test_property_impossible_support_is_preserved_exactly(seed: int) -> None:
    K = 1 + (seed % 3)
    D = 2 + (seed % 2)
    T = 2 + (seed % 4)
    dtype = torch.float64
    initial = torch.zeros(K, dtype=dtype).log_softmax(-1)
    duration = torch.full((T, K, D), float("-inf"), dtype=dtype)
    forced_duration_index = seed % D
    duration[..., forced_duration_index] = 0.0
    transition = torch.full((T, K, D, K), float("-inf"), dtype=dtype)
    forced_state = seed % K
    transition[..., forced_state] = 0.0
    emission = torch.zeros(T, K, dtype=dtype)

    expected = causal_prefix_joint_posterior(initial, duration, transition, emission)
    alpha = _forward(_model(K, D), _sequence(emission), initial, duration, transition)[0]
    z = torch.logsumexp(alpha.flatten(1), dim=-1)
    actual = alpha - z[:, None, None]
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected)), f"seed={seed}"


def test_property_self_transition_always_resets_episode_age() -> None:
    dtype = torch.float64
    for D in (1, 2, 3):
        T = 6
        initial = torch.zeros(1, dtype=dtype)
        duration = torch.full((T, 1, D), float("-inf"), dtype=dtype)
        duration[..., 0] = 0.0
        transition = torch.zeros(T, 1, D, 1, dtype=dtype)
        emission = torch.zeros(T, 1, dtype=dtype)
        alpha = _forward(_model(1, D), _sequence(emission), initial, duration, transition)[0]
        z = torch.logsumexp(alpha.flatten(1), dim=-1)
        normalized = alpha - z[:, None, None]
        assert torch.equal(normalized[:, 0, 0], torch.zeros(T, dtype=dtype))
        if D > 1:
            assert torch.isneginf(normalized[:, 0, 1:]).all()
