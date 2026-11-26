"""HSMM compatibility-boundary values and derivatives against probability products."""

import pytest
import torch
from nhsmm.filtering import HSMMFilterState
from nhsmm.survival import active_episode_survival_forecast
from nhsmm.transitions import one_step_transition_forecast

from tests.ml_stack.nhsmm.reference.test_hsmm_filtering_supervision import (
    branch_frontiers,
    expected_score,
)
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duration", [1, 2, 3])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("allow_self", [False, True])
def test_supervised_filter_boundary_all_parameter_gradients(duration, mixed, allow_self, dtype):
    from nhsmm import CausalContext

    model, x, context = fixture(duration, True, allow_self=allow_self)
    model.to(dtype=dtype)
    x = x.to(dtype=dtype)
    context = CausalContext(
        context.emission.to(dtype=dtype).detach().requires_grad_(),
        context.boundary.to(dtype=dtype).detach().requires_grad_(),
        context.available,
    )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    with torch.no_grad():
        model.dist.duration.tail_end_probability.copy_(x.new_tensor([1, 0.55 if mixed else 1, 1]))
    labels = torch.tensor([0, 0, 1, 2])
    reference = expected_score(branch_frontiers(parts(model, x, context)), labels)
    actual = model.supervised_filter_log_score(x, labels, context=context).sum()
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    parameters = [p for p in model.parameters() if p.requires_grad] + [
        context.emission,
        context.boundary,
    ]
    left = torch.autograd.grad(actual, parameters, allow_unused=True, retain_graph=True)
    right = torch.autograd.grad(reference, parameters, allow_unused=True)
    for parameter, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(parameter) if a is None else a
        b = torch.zeros_like(parameter) if b is None else b
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


def forecast_inputs(duration, dtype, batched_tail):
    logits = torch.linspace(-0.6, 0.8, 4 * duration, dtype=dtype).reshape(2, 2, duration)
    logits.requires_grad_()
    q = torch.linspace(0.4, -0.7, 4 * duration, dtype=dtype).reshape(2, 2, duration)
    q.requires_grad_()
    h = torch.tensor(
        [[1.0, 0.35], [0.6, 1.0]] if batched_tail else [1.0, 0.35], dtype=dtype, requires_grad=True
    )
    posterior = logits.flatten(1).log_softmax(-1).reshape_as(logits)
    return HSMMFilterState(posterior), q, h, logits


def survival_reference(state, scores, tail, horizons):
    """Direct conditional survival from PMF suffixes and geometric powers."""
    p, q = state.log_posterior.exp(), scores.softmax(-1)
    batch, states, duration = q.shape
    h = tail.expand(batch, states)
    rows = []
    for b in range(batch):
        columns = []
        for horizon in horizons:
            total = q.new_zeros(())
            for k in range(states):
                for a in range(duration):
                    exact = min(horizon, duration - 1 - a)
                    conditional = q[b, k, a + exact :].sum() / q[b, k, a:].sum()
                    conditional = conditional * (1 - h[b, k]) ** (horizon - exact)
                    total = total + p[b, k, a] * conditional
            columns.append(total)
        rows.append(torch.stack(columns))
    return torch.stack(rows)


@pytest.mark.parametrize("duration", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched_tail", [False, True])
def test_survival_boundary_values_first_and_second_derivatives(duration, dtype, batched_tail):
    state, q, h, logits = forecast_inputs(duration, dtype, batched_tail)
    horizons = [1, 2, 3, 7]
    result = active_episode_survival_forecast(state, q, horizons, h)
    expected = survival_reference(state, q, h, horizons)
    parameters = [h, q, logits]
    weights = q.new_tensor([[0.3, 0.8, -0.2, 1.1], [0.7, -0.4, 0.9, 0.1]])
    for actual, reference in (
        (result.survival_probability, expected),
        (result.end_within_probability, 1 - expected),
    ):
        tolerance = 3e-6 if dtype == torch.float32 else 1e-11
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
        ga = torch.autograd.grad(
            (actual * weights).sum(),
            parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        ge = torch.autograd.grad(
            (reference * weights).sum(),
            parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        for parameter, a, e in zip(parameters, ga, ge, strict=True):
            a = torch.zeros_like(parameter) if a is None else a
            e = torch.zeros_like(parameter) if e is None else e
            torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
        # Includes mixed tail/duration/posterior derivatives at exact h=1.
        ha = torch.autograd.grad(ga[0].sum(), parameters, retain_graph=True, allow_unused=True)
        he = torch.autograd.grad(ge[0].sum(), parameters, retain_graph=True, allow_unused=True)
        for parameter, a, e in zip(parameters, ha, he, strict=True):
            a = torch.zeros_like(parameter) if a is None else a
            e = torch.zeros_like(parameter) if e is None else e
            torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("duration", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched_tail", [False, True])
def test_next_state_prior_boundary_derivatives_and_conservation(duration, dtype, batched_tail):
    state, q, h, logits = forecast_inputs(duration, dtype, batched_tail)
    a = torch.linspace(-0.8, 0.5, 8 * duration, dtype=dtype).reshape(2, 2, duration, 2)
    a.requires_grad_()
    result = one_step_transition_forecast(state, q, a, h)
    p, mass, transition = state.log_posterior.exp(), q.softmax(-1), a.softmax(-1)
    hazards = torch.stack([mass[..., i] / mass[..., i:].sum(-1) for i in range(duration)], -1)
    hazards = torch.cat([hazards[..., :-1], h.expand(2, 2).unsqueeze(-1)], -1)
    joint = (p[..., None] * hazards[..., None] * transition).sum(2)
    prior = (p * (1 - hazards)).sum(-1) + joint.sum(1)
    parameters = [h, q, logits, a]
    tolerance = 3e-6 if dtype == torch.float32 else 1e-11
    for actual, expected in (
        (result.next_state_prior, prior),
        (result.boundary_transition_joint, joint),
    ):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        weights = torch.linspace(-0.7, 0.9, actual.numel(), dtype=dtype).reshape_as(actual)
        ga = torch.autograd.grad((actual * weights).sum(), parameters, retain_graph=True)
        ge = torch.autograd.grad((expected * weights).sum(), parameters, retain_graph=True)
        for g, e in zip(ga, ge, strict=True):
            torch.testing.assert_close(g, e, atol=tolerance, rtol=tolerance)
    conserved = torch.autograd.grad(result.next_state_prior.sum(), h)[0]
    torch.testing.assert_close(conserved, torch.zeros_like(h), atol=tolerance, rtol=0)


def test_filter_boundary_supports_second_derivatives():
    model, x, context = fixture(2, True)
    with torch.no_grad():
        model.dist.duration.tail_end_probability.fill_(1)
    score = model.supervised_filter_log_score(x, torch.tensor([0, 0, 1, 2]), context=context)
    gradient = torch.autograd.grad(
        score.sum(), model.dist.duration.tail_end_probability, create_graph=True
    )[0]
    curvature = torch.autograd.grad(gradient.sum(), model.dist.duration.tail_end_probability)[0]
    assert torch.isfinite(gradient).all() and torch.isfinite(curvature).all()


def test_filter_boundary_supervised_update_obeys_reference_direction():
    model, x, context = fixture(3, True)
    with torch.no_grad():
        model.dist.duration.tail_end_probability.fill_(1)
    labels = torch.tensor([0, 0, 1, 2])
    reference = expected_score(branch_frontiers(parts(model, x, context)), labels)
    gradient = torch.autograd.grad(reference, model.dist.duration.tail_end_probability)[0]
    assert bool((gradient[:2] > 0).all()) and gradient[2] < 0
    result = model.fit_supervised(
        x, labels, context=context, steps=1, lr=0.001, objective="causal_filter_log_score"
    )
    assert result["updates"] == 1
    h = model.dist.duration.tail_end_probability
    assert torch.equal(h[:2], torch.ones_like(h[:2]))
    assert h[2] < 1


def test_filter_boundary_ragged_empty_and_one_row_have_no_spurious_flux():
    from nhsmm import CausalContext

    model, x, context = fixture(2, True)
    with torch.no_grad():
        model.dist.duration.tail_end_probability.copy_(x.new_tensor([1, 0.55, 1]))
    lengths = [4, 2, 1, 0]
    observations = [x[:n] for n in lengths]
    labels = [torch.tensor([0, 0, 1, 2])[:n] for n in lengths]
    e = context.emission.unsqueeze(0).expand(4, -1, -1)
    u = context.boundary.unsqueeze(0).expand(4, -1, -1)
    available = context.available.unsqueeze(0).expand(4, -1)
    actual = model.supervised_filter_log_score(
        observations, labels, context=CausalContext(e, u, available)
    )
    expected = []
    for n, z in zip(lengths, labels, strict=True):
        if n:
            c = CausalContext(context.emission[:n], context.boundary[:n], context.available[:n])
            expected.append(expected_score(branch_frontiers(parts(model, x[:n], c)), z))
        else:
            expected.append(x.new_zeros(()))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    h = model.dist.duration.tail_end_probability
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), h, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), h)[0],
        atol=1e-10,
        rtol=1e-9,
    )


def test_filter_boundary_uses_each_distribution_head_once():
    from unittest.mock import patch

    model, x, context = fixture(2, True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.4
    with torch.no_grad():
        model.dist.duration.tail_end_probability.fill_(1)
    heads = [model.dist.initial, model.dist.duration, model.dist.transition]
    from contextlib import ExitStack

    with ExitStack() as stack:
        mocks = [
            stack.enter_context(patch.object(head, "log_matrix", wraps=head.log_matrix))
            for head in heads
        ]
        result = model.supervised_filter_log_score(x, torch.tensor([0, 0, 1, 2]), context=context)
    assert torch.isfinite(result).all()
    for mocked in mocks:
        assert mocked.call_count == 1
    gradient = torch.autograd.grad(result.sum(), model.dist.duration.tail_end_probability)[0]
    assert torch.isfinite(gradient).all()


def test_filter_boundary_gradient_matches_one_sided_numerical_difference():
    model, x, context = fixture(2, True)
    h = model.dist.duration.tail_end_probability
    with torch.no_grad():
        h.fill_(1)
    labels = torch.tensor([0, 0, 1, 2])

    def score():
        return model.supervised_filter_log_score(x, labels, context=context).sum()

    gradient = torch.autograd.grad(score(), h)[0]
    base = float(score().detach())
    step = 1e-5
    for index in range(3):
        values = []
        try:
            for offset in (1, 2):
                with torch.no_grad():
                    h[index] = 1 - offset * step
                values.append(float(score().detach()))
        finally:
            with torch.no_grad():
                h[index] = 1
        numerical = (3 * base - 4 * values[0] + values[1]) / (2 * step)
        assert float(gradient[index]) == pytest.approx(numerical, abs=2e-7, rel=2e-6)
