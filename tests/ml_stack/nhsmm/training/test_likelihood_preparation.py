"""Likelihood preparation preserves terminal evidence and its gradients."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from nhsmm import NHSMM, ModelConfig


def _causal_probability_evidence(model, sequence):
    """Independent episode-history products, not terminal forward reduction."""
    bsize, steps, states = sequence.log_probs.shape
    duration = model.config.max_duration
    pi = model.dist.initial.log_matrix(
        context=model._active_context(sequence.canonical), T=steps
    ).exp()
    q = model.dist.duration.log_matrix(
        context=model._active_context(sequence.contexts),
        T=steps,
        soft_dmax=model.duration_logits_bias,
    ).exp()
    a = model.dist.transition.log_matrix(
        context=model._active_context(sequence.contexts), T=steps
    ).exp()
    pi, q, a = (
        model._expand_batch(value, bsize, name)
        for value, name in ((pi, "initial"), (q, "duration"), (a, "transition"))
    )
    emission = sequence.log_probs.exp()
    tail = model.dist.duration.tail_probability()
    result = []
    for b, length in enumerate(sequence.lengths.tolist()):
        if not length:
            result.append(emission.new_zeros(()))
            continue
        paths = [(k, 1, pi[b, 0, k] * emission[b, 0, k]) for k in range(states)]
        for t in range(1, length):
            following = []
            for k, age, weight in paths:
                index = min(age, duration) - 1
                hazard = (
                    tail[k]
                    if tail is not None and age >= duration
                    else q[b, t - 1, k, index] / q[b, t - 1, k, index:].sum()
                )
                if tail is not None or age < duration:
                    following.append((k, age + 1, weight * (1 - hazard) * emission[b, t, k]))
                for dest in range(states):
                    following.append(
                        (
                            dest,
                            1,
                            weight * hazard * a[b, t - 1, k, index, dest] * emission[b, t, dest],
                        )
                    )
            paths = following
        result.append(torch.stack([weight for _, _, weight in paths]).sum().log())
    return torch.stack(result)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
def test_likelihood_prepares_once_and_preserves_evidence_gradients(causal, tail, reduce, ragged):
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=2,
            max_duration=3,
            causal=causal,
            duration_tail=tail,
            context_dim=3,
            seed=914,
            dropout=0.0,
        ),
        device="cpu",
    ).double()
    model.initialize_distributions()
    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    observations = [x[0], x[1, :2], x[1, :0]] if ragged else x
    context = torch.randn(len(observations), 5, 3, dtype=torch.float64, requires_grad=True)

    # Reduce terminal forward scores independently of the public likelihood.
    sequence = model._build_sequence_set(observations, context=context)
    alpha = model.forward(sequence)
    expected = torch.stack(
        [
            (
                torch.logsumexp(alpha[i, length - 1].reshape(-1), dim=0)
                if length
                else alpha.new_zeros(())
            )
            for i, length in enumerate(sequence.lengths.tolist())
        ]
    )
    if causal:
        expected = _causal_probability_evidence(model, sequence)
    if not causal:
        # This test checks one-time preparation and exact graph preservation.
        # Independent probability enumeration covers the partition's math.
        unit = replace(sequence, log_probs=torch.zeros_like(sequence.log_probs))
        partition_alpha = model.forward(unit)
        partition = torch.stack(
            [
                (
                    torch.logsumexp(partition_alpha[i, length - 1].reshape(-1), dim=0)
                    if length
                    else partition_alpha.new_zeros(())
                )
                for i, length in enumerate(sequence.lengths.tolist())
            ]
        )
        expected = expected - partition
    if reduce:
        expected = expected.sum()
    inputs = (x, context, *(p for p in model.parameters() if p.requires_grad))
    expected_grad = torch.autograd.grad(expected.sum(), inputs, allow_unused=True)

    with patch.object(model, "_ensure_tensor", wraps=model._ensure_tensor) as prepare:
        actual = model.log_likelihood(observations, context=context, reduce=reduce)
    assert prepare.call_count == 1
    # Centering changes causal floating-point addition order. Compare to
    # independent probability products within 64 Float64 eps, not bit identity
    # with the absolute-alpha reduction that loses small scores at large offsets.
    tolerance = 64 * torch.finfo(x.dtype).eps if causal else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    actual_grad = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    for actual_item, expected_item in zip(actual_grad, expected_grad):
        if expected_item is None:
            assert actual_item is None
        else:
            torch.testing.assert_close(actual_item, expected_item, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("observations", [[], torch.empty(0, 0, 2), torch.empty(2, 0, 2)])
@pytest.mark.parametrize("reduce", [False, True])
def test_empty_likelihood_prepares_once_and_remains_neutral(observations, reduce):
    model = NHSMM(ModelConfig(n_states=2, n_features=2, max_duration=3), device="cpu")
    model.initialize_distributions()
    with patch.object(model, "_ensure_tensor", wraps=model._ensure_tensor) as prepare:
        actual = model.log_likelihood(observations, reduce=reduce)
    assert prepare.call_count == 1
    expected = actual.new_zeros(()) if reduce else actual.new_zeros(len(observations))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
