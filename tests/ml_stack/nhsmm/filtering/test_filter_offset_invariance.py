"""Posterior, evidence and derivative invariance to shared emission offsets."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from nhsmm import CausalContext, HSMMFilterRuntime
from nhsmm.filtering import filter_model_sequence, filter_step, initialize_filter

from tests.ml_stack.nhsmm.filtering.test_viterbi_numerics import identical_emissions
from tests.ml_stack.nhsmm.reference.test_hsmm_filtering_supervision import (
    branch_frontiers,
    expected_score,
)
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


def probability_reference(model, length):
    """Independent expanded-state probability chain; identical emissions cancel."""
    d = model.config.max_duration
    pi = model.dist.initial.logits.detach().double().numpy()
    pi = np.exp(pi - pi.max())
    pi /= pi.sum()
    a = np.array([[0.9, 0.1], [0.1, 0.9]])
    matrix = np.zeros((2 * d, 2 * d))
    for state in range(2):
        for age in range(d):
            h = 0.25 if model.config.duration_tail and age == d - 1 else 1 / (d - age)
            for dest in range(2):
                matrix[state * d + age, dest * d] += h * a[state, dest]
            if age < d - 1 or model.config.duration_tail:
                matrix[state * d + age, state * d + min(age + 1, d - 1)] += 1 - h
    value = np.zeros(2 * d)
    value[::d] = pi
    rows = [value.reshape(2, d).copy()]
    for _ in range(1, length):
        value = value @ matrix
        rows.append(value.reshape(2, d).copy())
    return torch.tensor(np.stack(rows), dtype=next(model.parameters()).dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duration,tail", [(1, False), (3, False), (1, True), (3, True)])
def test_identical_large_gaussians_preserve_ragged_filter_evidence_and_gradients(
    dtype, duration, tail
):
    model = identical_emissions(dtype, True, duration, tail)
    expected = probability_reference(model, 4)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    large = 1e4 if dtype == torch.float32 else 1e9
    labels = [torch.ones(n, dtype=torch.long) for n in (4, 1, 0)]
    for value in (large, -large):
        x = torch.full((4, 1), value, dtype=dtype)
        xs = [x, x[:1], x[:0]]
        trace = filter_model_sequence(model, xs)
        torch.testing.assert_close(
            trace.log_posterior[0].exp(), expected, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            trace.log_posterior[1, 0].exp(), expected[0], atol=tolerance, rtol=tolerance
        )
        assert torch.isneginf(trace.log_posterior[2]).all()
        runtime = HSMMFilterRuntime(model)
        for t, row in enumerate(x):
            actual = runtime.step(row)
            torch.testing.assert_close(
                actual.log_posterior[0].exp(), expected[t], atol=tolerance, rtol=tolerance
            )
        target = expected[:, 1].sum(-1).log().sum() + expected[0, 1].sum().log()
        score = model.supervised_filter_log_score(xs, labels, reduce=True)
        torch.testing.assert_close(score, target, atol=5 * tolerance, rtol=tolerance)
        evidence = model.log_likelihood(xs)
        wanted = torch.tensor(
            [-n * (0.5 * value**2 + 0.5 * math.log(2 * math.pi)) for n in (4, 1, 0)],
            dtype=dtype,
        )
        torch.testing.assert_close(evidence, wanted, atol=0, rtol=8 * torch.finfo(dtype).eps)
        # With identical emissions, evidence is independent of all latent laws.
        parameters = [
            model.dist.initial.logits,
            model.dist.transition.logits,
            model.dist.duration.logits,
        ]
        if tail:
            parameters.append(model.dist.duration.tail_end_probability)
        gradients = torch.autograd.grad(evidence.sum(), parameters, allow_unused=True)
        for p, gradient in zip(parameters, gradients, strict=True):
            if gradient is not None:
                torch.testing.assert_close(
                    gradient, torch.zeros_like(p), atol=5 * tolerance, rtol=0
                )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "duration,tail,boundary",
    [
        (1, False, False),
        (2, False, False),
        (1, True, False),
        (3, True, False),
        (2, True, True),
    ],
)
@pytest.mark.parametrize("filtering", [False, True])
def test_offset_score_and_all_parameter_context_gradients_match_enumeration(
    dtype, duration, tail, boundary, filtering
):
    model, x, context = fixture(duration, tail, first_available=False, allow_self=False)
    model.to(dtype=dtype)
    x = x.to(dtype)
    c = CausalContext(
        context.emission.detach().to(dtype).requires_grad_(),
        context.boundary.detach().to(dtype).requires_grad_(),
        context.available,
    )
    if boundary:
        with torch.no_grad():
            model.dist.duration.tail_end_probability[1] = 1
    relative = torch.tensor(
        [[0, -8, -16], [-8, 0, -8], [0, 0, -8], [-16, -8, 0]], dtype=dtype, requires_grad=True
    )
    ingredients = list(parts(model, x, c))
    ingredients[1] = relative.exp()
    frontiers = branch_frontiers(tuple(ingredients))
    labels = torch.tensor([0, 0, 1, 2])
    if filtering:
        reference = expected_score(frontiers, labels)
    else:
        # Independent probability-space enumeration with zero-weight branches
        # retained at h=1, so the feasible one-sided derivative is represented.
        from tests.ml_stack.nhsmm.reference.neural_parameter_reference import enumerate_weights

        reference = (
            torch.stack(
                [
                    w
                    for _, w, _ in enumerate_weights(
                        tuple(ingredients), hmm=duration == 1 and not tail, length=4
                    )
                ]
            )
            .sum()
            .log()
        )
    offset = -(2**24 if dtype == torch.float32 else 2**52)
    data = model._build_sequence_set(x, context=c)
    data.log_probs = (relative + offset).unsqueeze(0)
    with patch.object(model, "_build_sequence_set", return_value=data):
        actual = (
            model.supervised_filter_log_score(x, labels, context=c).sum()
            if filtering
            else model.log_likelihood(x, context=c).sum()
        )
    wanted = reference if filtering else reference + 4 * offset
    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, wanted, atol=tolerance, rtol=8 * torch.finfo(dtype).eps)
    parameters = [p for p in model.parameters() if p.requires_grad] + [
        relative,
        c.emission,
        c.boundary,
    ]
    left = torch.autograd.grad(actual, parameters, allow_unused=True, retain_graph=True)
    right = torch.autograd.grad(reference, parameters, allow_unused=True)
    for p, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        assert torch.isfinite(a).all()
        torch.testing.assert_close(
            a, b, atol=tolerance, rtol=3e-4 if dtype == torch.float32 else 2e-8
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1, 1])
def test_public_transition_score_normalization_preserves_different_support_sizes(dtype, sign):
    offset = sign * (2**24 if dtype == torch.float32 else 2**52)
    initial = torch.zeros(2, dtype=dtype)
    state = initialize_filter(initial, torch.full_like(initial, offset), 1)
    a = torch.tensor([[[0.0, -torch.inf]], [[0.0, 0.0]]], dtype=dtype) + offset
    result = filter_step(
        state, torch.full_like(initial, offset), torch.full((2, 1), offset, dtype=dtype), a
    )
    torch.testing.assert_close(
        result.state_posterior[0],
        torch.tensor([0.75, 0.25], dtype=dtype),
        atol=2e-7 if dtype == torch.float32 else 2e-15,
        rtol=0,
    )
