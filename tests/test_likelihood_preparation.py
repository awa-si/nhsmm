"""Likelihood preparation preserves terminal evidence and its gradients."""

from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


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
    if reduce:
        expected = expected.sum()
    inputs = (x, context, *(p for p in model.parameters() if p.requires_grad))
    expected_grad = torch.autograd.grad(expected.sum(), inputs, allow_unused=True)

    with patch.object(model, "_ensure_tensor", wraps=model._ensure_tensor) as prepare:
        actual = model.log_likelihood(observations, context=context, reduce=reduce)
    assert prepare.call_count == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
    for actual_item, expected_item in zip(actual_grad, expected_grad):
        if expected_item is None:
            assert actual_item is None
        else:
            torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


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
