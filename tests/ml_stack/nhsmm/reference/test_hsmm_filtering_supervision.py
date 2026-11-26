"""HSMM prefix supervision against independently enumerated episode branches."""

import pytest
import torch
from nhsmm import CausalContext
from nhsmm.context import ContextRouter
from nhsmm.models.forward import _forward_causal_hazard

from .test_neural_parameter_math import fixture, parts

CASES = [(2, False), (3, False), (1, True), (2, True), (3, True)]


def branch_frontiers(ingredients):
    """Enumerate individual reset/continuation histories in probability space.

    No owner hazards, distributions, filtering or transition kernels are called.
    A row stores the uncapped true age, including tails beyond D.
    """
    pi, emission, q, transition, tail = ingredients
    k, d = q.shape[1:]
    rows = [(s, 1, pi[s] * emission[0, s]) for s in range(k)]
    frontiers = []
    for t in range(len(emission)):
        if t:
            following = []
            for s, age, weight in rows:
                i = min(age, d) - 1
                end = (
                    tail[s]
                    if tail is not None and age >= d
                    else q[t - 1, s, i] / q[t - 1, s, i:].sum()
                )
                if tail is not None or age < d:
                    following.append((s, age + 1, weight * (1 - end) * emission[t, s]))
                for j in range(k):
                    following.append(
                        (j, 1, weight * end * transition[t - 1, s, i, j] * emission[t, j])
                    )
            rows = following
        frontier = torch.stack(
            [
                torch.stack(
                    [
                        sum(
                            (w for s, age, w in rows if s == state and min(age, d) == bucket + 1),
                            pi.new_zeros(()),
                        )
                        for bucket in range(d)
                    ]
                )
                for state in range(k)
            ]
        )
        frontiers.append(frontier / frontier.sum())
    return torch.stack(frontiers)


def expected_score(frontiers, labels):
    p = frontiers.sum(-1)
    return p[torch.arange(len(labels)), labels].log().sum()


@pytest.mark.parametrize("duration,tail", CASES)
@pytest.mark.parametrize("allow_self", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_prefix_probabilities_and_all_raw_parameter_context_gradients(
    duration, tail, allow_self, available
):
    model, x, c = fixture(duration, tail, available, allow_self=allow_self)
    labels = torch.tensor([0, 0, 1, 2])
    reference = branch_frontiers(parts(model, x, c))
    expected = expected_score(reference, labels)
    actual = model.supervised_filter_log_score(x, labels, context=c).sum()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    data = model._build_sequence_set(x, context=c)
    actual_frontier = _forward_causal_hazard(
        model, data, ContextRouter.from_tensor(data), normalized=True
    )[0]
    torch.testing.assert_close(actual_frontier.exp(), reference, atol=1e-12, rtol=1e-10)
    parameters = [p for p in model.parameters() if p.requires_grad] + [c.emission, c.boundary]
    left = torch.autograd.grad(actual, parameters, allow_unused=True, retain_graph=True)
    right = torch.autograd.grad(expected, parameters, allow_unused=True, retain_graph=True)
    for p, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    assert torch.equal(left[-2][~c.available], torch.zeros_like(left[-2][~c.available]))
    assert torch.equal(left[-1][~c.available], torch.zeros_like(left[-1][~c.available]))
    assert torch.equal(left[-1][-1], torch.zeros_like(left[-1][-1]))
    for n in range(1, len(x) + 1):
        prefix = CausalContext(c.emission[:n], c.boundary[:n], c.available[:n])
        torch.testing.assert_close(
            model.supervised_filter_log_score(x[:n], labels[:n], context=prefix).sum(),
            expected_score(reference[:n], labels[:n]),
            atol=1e-12,
            rtol=1e-10,
        )


@pytest.mark.parametrize("duration,tail", CASES)
def test_finite_difference_duration_transition_tail_and_context(duration, tail):
    model, x, c = fixture(duration, tail)
    labels = torch.tensor([0, 0, 1, 2])
    probes = [
        (model.duration_logits_bias, 0),
        (model.dist.duration.logits, 0),
        (model.dist.transition.logits, 1),
        (model.dist.duration.context_net[-1].weight, 0),
        (model.dist.emission.context_net[-1].bias, 0),
        (c.emission, 0),
        (c.boundary, 0),
    ]
    if tail:
        probes.append((model.dist.duration.tail_end_probability, 0))

    def score():
        return model.supervised_filter_log_score(x, labels, context=c).sum()

    gradients = torch.autograd.grad(score(), [p for p, _ in probes], allow_unused=True)
    for (p, index), g in zip(probes, gradients, strict=True):
        original = float(p.detach().flatten()[index])
        values = []
        try:
            for sign in (1, -1):
                with torch.no_grad():
                    p.view(-1)[index] = original + sign * 2e-5
                values.append(float(score().detach()))
        finally:
            with torch.no_grad():
                p.view(-1)[index] = original
        derivative = float(g.flatten()[index]) if g is not None else 0.0
        assert derivative == pytest.approx((values[0] - values[1]) / 4e-5, abs=2e-5, rel=3e-4)


@pytest.mark.parametrize("duration,tail", CASES)
def test_ragged_empty_resets_no_teacher_forcing_and_fit(duration, tail):
    model, x, _ = fixture(duration, tail)
    labels = torch.tensor([0, 0, 1, 2])
    xs, zs = [x, x[:2], x[:0]], [labels, labels[:2], labels[:0]]
    batch = model.supervised_filter_log_score(xs, zs)
    separate = torch.cat([model.supervised_filter_log_score(a, b) for a, b in zip(xs, zs)])
    torch.testing.assert_close(batch, separate, atol=1e-12, rtol=1e-10)
    assert batch[-1] == 0
    changed = labels.clone()
    changed[0] = 2
    torch.testing.assert_close(
        model.supervised_filter_log_score(x, labels)
        - model.supervised_filter_log_score(x, changed),
        model.supervised_filter_log_score(x[:1], labels[:1])
        - model.supervised_filter_log_score(x[:1], changed[:1]),
        atol=1e-12,
        rtol=1e-10,
    )
    result = model.fit_supervised(xs, zs, steps=4, lr=0.01, objective="causal_filter_log_score")
    assert result["updates"] == 4 and result["rows"] == 6
    assert result["final_nll_per_row"] < result["initial_nll_per_row"]


@pytest.mark.parametrize("tail", [False, True])
def test_one_row_has_no_duration_exit_or_boundary_derivative(tail):
    model, x, c = fixture(3, tail)
    z = torch.tensor([0])
    score = model.supervised_filter_log_score(
        x[:1], z, context=CausalContext(c.emission[:1], c.boundary[:1], c.available[:1])
    ).sum()
    parameters = (
        list(model.dist.duration.parameters())
        + list(model.dist.transition.parameters())
        + [model.duration_logits_bias, c.boundary]
    )
    gradients = torch.autograd.grad(
        score, [p for p in parameters if p.requires_grad], allow_unused=True
    )
    assert all(g is None or torch.equal(g, torch.zeros_like(g)) for g in gradients)


@pytest.mark.parametrize("tail", [False, True])
def test_structural_zero_score_and_finite_supported_gradient(tail, monkeypatch):
    model, x, c = fixture(2, tail)
    # Deterministic age-two duration: no reset is allowed at the first boundary.
    monkeypatch.setattr(
        model.dist.duration,
        "log_matrix",
        lambda **kw: x.new_tensor([[[[-torch.inf, 0.0]] * 3] * 4]),
    )
    monkeypatch.setattr(
        model.dist.initial,
        "log_matrix",
        lambda **kw: x.new_tensor([[[0.0, -torch.inf, -torch.inf]]]),
    )
    z = torch.tensor([0, 0, 1, 1])
    score = model.supervised_filter_log_score(x, z, context=c).sum()
    assert torch.isfinite(score)
    grads = torch.autograd.grad(
        score, [p for p in model.parameters() if p.requires_grad], allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    z[1] = 1
    assert torch.isneginf(model.supervised_filter_log_score(x, z, context=c)).all()
    with pytest.raises(ValueError, match="non-finite"):
        model.fit_supervised(x, z, context=c, steps=1, objective="causal_filter_log_score")
