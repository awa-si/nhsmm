"""End-to-end numerical verification from raw parameters, no owner scores in oracle."""

import ast
import math
from pathlib import Path

import pytest
import torch
from nhsmm import NHSMM, CausalContext, HSMMFilterRuntime, ModelConfig
from nhsmm.filtering import filter_model_sequence

from .neural_parameter_reference import enumerate_weights, ingredients, path_weight

CASES = [(1, False), (2, False), (1, True), (2, True)]


def fixture(duration, tail, first_available=True, *, allow_self=None):
    model = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=duration,
            duration_tail=tail,
            allow_self_transitions=(
                (duration == 1 and not tail) if allow_self is None else allow_self
            ),
            causal=True,
            context_dim=2,
            hidden_dim=4,
            activation="tanh",
            use_context_encoder=False,
            seed=713,
            dropout=0,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.double().eval()
    with torch.no_grad():
        model.dist.initial.logits.copy_(torch.tensor([0.4, -0.3, 0.2]))
        model.dist.duration.logits.copy_(
            torch.linspace(-0.5, 0.6, 3 * duration).reshape(3, duration)
        )
        model.dist.transition.logits.copy_(
            torch.linspace(-0.7, 0.8, 9 * duration).reshape(3, duration, 3)
        )
        model.dist.emission.mu.copy_(torch.tensor([[-0.8, 0.4], [0.1, -0.2], [0.9, 0.6]]))
        model.dist.emission.log_var.copy_(torch.tensor([[-0.6, 0.2], [0.3, -0.4], [0.1, 0.6]]))
        model.duration_logits_bias.copy_(
            torch.linspace(-0.9, 0.5, 3 * duration).reshape(3, duration)
        )
        for component in (
            model.dist.initial,
            model.dist.duration,
            model.dist.transition,
            model.dist.emission,
        ):
            component.delta_scale.fill_(0.7)
            component.log_temperature.fill_(math.log(1.3))
        if tail:
            model.dist.duration.tail_end_probability.copy_(torch.tensor([0.2, 0.55, 0.8]))
    x = torch.tensor([[-0.3, 0.1], [0.5, -0.4], [0.2, 0.7], [-0.5, 0.3]], dtype=torch.float64)
    e = torch.tensor(
        [[0.2, -0.4], [0.7, 0.5], [-0.3, 0.6], [0.1, -0.8]], dtype=x.dtype, requires_grad=True
    )
    u = torch.tensor(
        [[-0.5, 0.3], [0.6, -0.2], [0.4, 0.9], [-0.7, -0.1]], dtype=x.dtype, requires_grad=True
    )
    mask = torch.tensor([first_available, not first_available, True, True])
    return model, x, CausalContext(e, u, mask)


def parts(model, x, context):
    parameters = {
        name: dict(getattr(model.dist, name).named_parameters())
        for name in ("initial", "duration", "transition", "emission")
    }
    return ingredients(
        parameters,
        x,
        context.emission,
        context.boundary,
        context.available,
        model.duration_logits_bias,
        model.config.max_duration,
        model.config.duration_tail,
        model.config.allow_self_transitions,
    )


@pytest.mark.parametrize("duration,tail", CASES)
@pytest.mark.parametrize("first_available", [False, True])
def test_joint_marginal_filter_map_and_runtime_from_raw_weights(duration, tail, first_available):
    model, x, context = fixture(duration, tail, first_available)
    reference = parts(model, x, context)
    hmm = duration == 1 and not tail
    trace = filter_model_sequence(model, x, context=context)
    runtime = HSMMFilterRuntime(model)
    for length in range(1, len(x) + 1):
        rows = enumerate_weights(reference, hmm=hmm, length=length)
        total = torch.stack([weight for _, weight, _ in rows]).sum()
        frontier = x.new_zeros(3, duration)
        for path, weight, age in rows:
            if age <= duration or tail:
                frontier[path[-1], min(age, duration) - 1] += weight
        torch.testing.assert_close(
            trace.log_posterior[0, length - 1].exp(), frontier / total, atol=1e-12, rtol=1e-10
        )
        prefix_context = CausalContext(
            context.emission[:length], context.boundary[:length], context.available[:length]
        )
        torch.testing.assert_close(
            model.log_likelihood(x[:length], context=prefix_context)[0],
            total.log(),
            atol=1e-12,
            rtol=1e-10,
        )
        state = runtime.step(
            x[length - 1],
            context=CausalContext(
                context.emission[length - 1],
                context.boundary[length - 1],
                context.available[length - 1],
            ),
        )
        torch.testing.assert_close(
            state.log_posterior[0].exp(), frontier / total, atol=1e-12, rtol=1e-10
        )
    for path, weight, _ in rows:
        actual = model.supervised_log_likelihood(x, torch.tensor(path), context=context)[0]
        torch.testing.assert_close(actual.exp(), weight, atol=1e-12, rtol=1e-10)
    winner = max(rows, key=lambda row: float(row[1].detach()))[0]
    assert tuple(model.predict(x, context=context, verbose=False)[0].tolist()) == winner


@pytest.mark.parametrize("duration,tail", CASES)
def test_all_parameter_and_context_gradients_match_independent_products(duration, tail):
    model, x, context = fixture(duration, tail)
    reference = parts(model, x, context)
    hmm = duration == 1 and not tail
    labels = torch.tensor([0, 0, 1, 2])
    joint = path_weight(labels.tolist(), *reference, hmm=hmm)[0].log()
    marginal = (
        torch.stack([weight for _, weight, _ in enumerate_weights(reference, hmm=hmm, length=4)])
        .sum()
        .log()
    )
    parameters = [p for p in model.parameters() if p.requires_grad] + [
        context.emission,
        context.boundary,
    ]
    for actual, expected in (
        (model.supervised_log_likelihood(x, labels, context=context).sum(), joint),
        (model.log_likelihood(x, context=context).sum(), marginal),
    ):
        left = torch.autograd.grad(actual, parameters, allow_unused=True, retain_graph=True)
        right = torch.autograd.grad(expected, parameters, allow_unused=True, retain_graph=True)
        for parameter, a, b in zip(parameters, left, right, strict=True):
            a = torch.zeros_like(parameter) if a is None else a
            b = torch.zeros_like(parameter) if b is None else b
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        assert torch.equal(
            left[-2][~context.available], torch.zeros_like(left[-2][~context.available])
        )
        assert torch.equal(
            left[-1][~context.available], torch.zeros_like(left[-1][~context.available])
        )


def test_neural_tail_public_score_derivatives_match_central_difference():
    model, x, context = fixture(2, True)
    labels = torch.tensor([0, 0, 1, 2])
    probes = [
        (model.dist.emission.context_net[-1].bias, 0),
        (model.dist.duration.context_net[-1].weight, 0),
        (model.dist.transition.context_net[-1].bias, 1),
        (model.dist.duration.tail_end_probability, 0),
        (model.duration_logits_bias, 0),
    ]
    for supervised in (False, True):

        def score():
            return (
                model.supervised_log_likelihood(x, labels, context=context)
                if supervised
                else model.log_likelihood(x, context=context)
            ).sum()

        gradients = torch.autograd.grad(score(), [p for p, _ in probes], allow_unused=True)
        for (parameter, index), gradient in zip(probes, gradients, strict=True):
            original = float(parameter.detach().flatten()[index])
            values = []
            try:
                for sign in (1, -1):
                    with torch.no_grad():
                        parameter.view(-1)[index] = original + sign * 2e-5
                    values.append(float(score().detach()))
            finally:
                with torch.no_grad():
                    parameter.view(-1)[index] = original
            expected = (values[0] - values[1]) / 4e-5
            actual = float(gradient.flatten()[index]) if gradient is not None else 0.0
            assert actual == pytest.approx(expected, abs=2e-5, rel=3e-4)


def test_parameter_oracle_never_imports_production_or_scores():
    tree = ast.parse(Path(__file__).with_name("neural_parameter_reference.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("nhsmm") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("nhsmm")
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"log_matrix", "log_prob", "_build_sequence_set", "filter_step"}
