from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet


def _normalized_scores(K: int, D: int, T: int, dtype: torch.dtype, seed: int = 8801):
    g = torch.Generator().manual_seed(seed)
    initial = torch.randn(K, generator=g, dtype=dtype).log_softmax(-1)
    duration = torch.randn(T, K, D, generator=g, dtype=dtype).log_softmax(-1)
    transition = torch.randn(T, K, D, K, generator=g, dtype=dtype).log_softmax(-1)
    emission = torch.randn(T, K, generator=g, dtype=dtype) - 0.2
    return initial, duration, transition, emission


def _sequence(emission: torch.Tensor, length: int | None = None) -> SequenceSet:
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


def _model(K: int, D: int, causal: bool = True) -> NHSMM:
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


def _forward_with_scores(model, sequence, initial, duration, transition):
    K = initial.shape[-1]
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        return model.forward(sequence)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_k1_has_no_state_uncertainty(dtype: torch.dtype) -> None:
    K, D, T = 1, 3, 5
    initial, duration, transition, emission = _normalized_scores(K, D, T, dtype)
    alpha = _forward_with_scores(_model(K, D), _sequence(emission), initial, duration, transition)
    normalized = alpha[0] - torch.logsumexp(alpha[0].flatten(1), dim=-1)[:, None, None]
    state_posterior = torch.logsumexp(normalized, dim=-1).exp()
    torch.testing.assert_close(state_posterior, torch.ones(T, 1, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_d1_forces_every_timestep_to_age_one(dtype: torch.dtype) -> None:
    K, D, T = 3, 1, 5
    initial, duration, transition, emission = _normalized_scores(K, D, T, dtype)
    alpha = _forward_with_scores(_model(K, D), _sequence(emission), initial, duration, transition)
    assert alpha.shape[-1] == 1
    normalized = alpha[0] - torch.logsumexp(alpha[0].flatten(1), dim=-1)[:, None, None]
    age_posterior = torch.logsumexp(normalized, dim=1).exp()
    torch.testing.assert_close(age_posterior, torch.ones(T, 1, dtype=dtype))


@pytest.mark.parametrize("causal", [False, True])
def test_t1_uses_only_initial_and_first_emission(causal: bool) -> None:
    K, D, T = 3, 3, 1
    dtype = torch.float64
    initial, duration, transition, emission = _normalized_scores(K, D, T, dtype)
    alpha = _forward_with_scores(
        _model(K, D, causal=causal), _sequence(emission), initial, duration, transition
    )
    actual = torch.logsumexp(alpha[0, 0].flatten(), dim=0)
    if causal:
        expected = torch.logsumexp(initial + emission[0], dim=0)
    else:
        expected = torch.logsumexp(initial + duration[0, :, 0] + emission[0], dim=0)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_causal_self_transition_resets_age_to_one() -> None:
    K, D, T = 1, 2, 3
    dtype = torch.float64
    initial = torch.zeros(K, dtype=dtype)
    duration = torch.full((T, K, D), float("-inf"), dtype=dtype)
    duration[..., 0] = 0.0
    transition = torch.zeros(T, K, D, K, dtype=dtype)
    emission = torch.zeros(T, K, dtype=dtype)
    alpha = _forward_with_scores(_model(K, D), _sequence(emission), initial, duration, transition)
    normalized = alpha[0] - torch.logsumexp(alpha[0].flatten(1), dim=-1)[:, None, None]
    assert torch.equal(normalized[:, 0, 0], torch.zeros(T, dtype=dtype))
    assert torch.isneginf(normalized[:, 0, 1]).all()


def test_impossible_support_remains_exact_negative_infinity() -> None:
    K, D, T = 1, 2, 2
    dtype = torch.float64
    initial = torch.zeros(K, dtype=dtype)
    duration = torch.full((T, K, D), float("-inf"), dtype=dtype)
    duration[..., 1] = 0.0
    transition = torch.zeros(T, K, D, K, dtype=dtype)
    emission = torch.zeros(T, K, dtype=dtype)
    alpha = _forward_with_scores(_model(K, D), _sequence(emission), initial, duration, transition)
    normalized = alpha[0] - torch.logsumexp(alpha[0].flatten(1), dim=-1)[:, None, None]
    assert torch.isneginf(normalized[0, 0, 1])
    assert torch.isneginf(normalized[1, 0, 0])
    assert normalized[0, 0, 0].item() == 0.0
    assert normalized[1, 0, 1].item() == 0.0


def test_right_padding_cannot_change_valid_prefix() -> None:
    K, D, T, L = 2, 3, 6, 3
    dtype = torch.float64
    initial, duration, transition, emission = _normalized_scores(K, D, T, dtype)
    padded_a = emission.clone()
    padded_b = emission.clone()
    padded_b[L:] = torch.tensor([[100.0, -100.0], [-50.0, 50.0], [80.0, -80.0]], dtype=dtype)
    model = _model(K, D)
    alpha_a = _forward_with_scores(model, _sequence(padded_a, L), initial, duration, transition)
    alpha_b = _forward_with_scores(model, _sequence(padded_b, L), initial, duration, transition)
    torch.testing.assert_close(alpha_a[:, :L], alpha_b[:, :L], atol=0.0, rtol=0.0)


def test_batch_result_matches_single_sequence_results() -> None:
    K, D, T = 2, 3, 5
    dtype = torch.float64
    lengths = [5, 3]
    items = [_normalized_scores(K, D, T, dtype, seed=8900 + i) for i in range(2)]
    initial = torch.stack([x[0] for x in items])
    duration = torch.stack([x[1] for x in items])
    transition = torch.stack([x[2] for x in items])
    emission = torch.stack([x[3] for x in items])
    mask = (torch.arange(T).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)).unsqueeze(-1)
    sequence = SequenceSet(
        sequences=torch.zeros(2, T, 1, dtype=dtype),
        lengths=torch.tensor(lengths),
        masks=mask,
        contexts=torch.zeros(2, T, 0, dtype=dtype),
        canonical=torch.zeros(2, 1, 0, dtype=dtype),
        log_probs=emission,
    )
    model = _model(K, D)
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.unsqueeze(1)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration),
        patch.object(model.dist.transition, "log_matrix", return_value=transition),
    ):
        batch = model.forward(sequence)

    for b, L in enumerate(lengths):
        single = _forward_with_scores(
            model,
            _sequence(emission[b], L),
            initial[b],
            duration[b],
            transition[b],
        )
        torch.testing.assert_close(batch[b, :L], single[0, :L], atol=2e-12, rtol=2e-12)


def test_causal_prefix_is_invariant_to_future_observation_changes() -> None:
    K, D, T, prefix = 2, 3, 6, 3
    dtype = torch.float64
    initial, duration, transition, emission = _normalized_scores(K, D, T, dtype)
    changed = emission.clone()
    changed[prefix:] += torch.tensor([[20.0, -20.0], [-15.0, 15.0], [30.0, -30.0]], dtype=dtype)
    model = _model(K, D)
    first = _forward_with_scores(model, _sequence(emission), initial, duration, transition)
    second = _forward_with_scores(model, _sequence(changed), initial, duration, transition)
    torch.testing.assert_close(first[:, :prefix], second[:, :prefix], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_causal_posterior_normalizes_at_every_timestep(dtype: torch.dtype) -> None:
    K, D, T = 3, 3, 6
    initial, duration, transition, emission = _normalized_scores(K, D, T, dtype)
    alpha = _forward_with_scores(_model(K, D), _sequence(emission), initial, duration, transition)
    normalized = alpha[0] - torch.logsumexp(alpha[0].flatten(1), dim=-1)[:, None, None]
    probability_mass = normalized.exp().sum(dim=(1, 2))
    tol = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(probability_mass, torch.ones(T, dtype=dtype), atol=tol, rtol=tol)
    assert int((normalized.exp() > 0).nonzero()[:, 2].max().item()) + 1 <= D
