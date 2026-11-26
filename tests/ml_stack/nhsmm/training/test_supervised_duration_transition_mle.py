"""Supervised duration/transition sufficient-statistic and MLE references."""

import math

import numpy as np
import pytest
import torch
from nhsmm import NHSMM, ModelConfig

from tests.ml_stack.nhsmm.reference.neural_parameter_reference import path_weight
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


def sample_paths(D, tail, *, seed=815):
    """Independent state/true-age generator; all last episodes remain censored."""
    rng = np.random.default_rng(seed)
    q = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])[:, :D]
    q /= q.sum(-1, keepdims=True)
    h_tail = np.array([0.25, 0.6, 0.4])
    A = np.zeros((3, D, 3))
    for k in range(3):
        for a in range(D):
            destinations = (
                list(range(3)) if D == 1 and not tail else [j for j in range(3) if j != k]
            )
            weights = rng.uniform(0.2, 1.0, len(destinations))
            A[k, a, destinations] = weights / weights.sum()
    paths = []
    for length in range(48, 108, 5):
        state, age = int(rng.integers(3)), 1
        z = [state]
        for _ in range(length - 1):
            index = min(age, D) - 1
            h = h_tail[state] if tail and age >= D else q[state, index] / q[state, index:].sum()
            if rng.random() < h:
                state = int(rng.choice(3, p=A[state, index]))
                age = 1
            else:
                age += 1
            z.append(state)
        paths.append(torch.tensor(z, dtype=torch.long))
    return paths


def counts(paths, D, tail):
    risk = np.zeros((3, D), dtype=np.int64)
    end = np.zeros_like(risk)
    transitions = np.zeros((3, D, 3), dtype=np.int64)
    hmm = D == 1 and not tail
    for z in paths:
        age = 1
        for left, right in zip(z.tolist()[:-1], z.tolist()[1:]):
            a = min(age, D) - 1
            risk[left, a] += 1
            if hmm or left != right:
                end[left, a] += 1
                transitions[left, a, right] += 1
                age = 1
            else:
                age += 1
    return risk, end, transitions


def event_score(q, transition, h_tail, risk, end, transitions):
    """Log probability from aggregate observed edges, excluding all final exits."""
    value = q.new_zeros(())
    for k in range(3):
        for a in range(q.shape[-1]):
            hazard = (
                h_tail[k]
                if h_tail is not None and a == q.shape[-1] - 1
                else (q[k, a] / q[k, a:].sum())
            )
            if end[k, a]:
                value = value + int(end[k, a]) * hazard.log()
            if risk[k, a] > end[k, a]:
                value = value + int(risk[k, a] - end[k, a]) * torch.log1p(-hazard)
            for j in range(3):
                if transitions[k, a, j]:
                    value = value + int(transitions[k, a, j]) * transition[k, a, j].log()
    return value


def make_model(D, tail):
    m = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=1,
            max_duration=D,
            duration_tail=tail,
            causal=True,
            allow_self_transitions=(D == 1 and not tail),
            seed=815,
            initial_init_mode="uniform",
            duration_init_mode="uniform",
            transition_init_mode="uniform",
            dropout=0,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    m.double().eval()
    with torch.no_grad():
        m.dist.initial.logits.zero_()
        m.dist.emission.mu.zero_()
        m.dist.emission.log_var.fill_(math.log(math.expm1(1.0)))
    for p in m.parameters():
        p.requires_grad_(False)
    m.dist.duration.logits.requires_grad_()
    m.dist.transition.logits.requires_grad_()
    if tail:
        m.dist.duration.tail_end_probability.requires_grad_()
    return m


def probabilities(m):
    q = m.dist.duration.logits.softmax(-1)  # frozen, equal soft gates cancel
    logits = m.dist.transition.logits
    if not m.config.allow_self_transitions:
        logits = logits.masked_fill(torch.eye(3, dtype=torch.bool)[:, None, :], -torch.inf)
    return q, logits.softmax(-1), m.dist.duration.tail_end_probability


@pytest.mark.parametrize("D,tail", [(1, False), (3, False), (3, True)])
def test_supervised_fit_recovers_censored_duration_and_transition_mle(D, tail, record_property):
    paths = sample_paths(D, tail)
    xs = [torch.zeros(len(z), 1, dtype=torch.float64) for z in paths]
    risk, end, transitions = counts(paths, D, tail)
    assert np.all(risk > 0) and np.all(transitions.sum(-1) > 0)
    assert int(risk.sum()) == sum(len(z) - 1 for z in paths)
    m = make_model(D, tail)
    rows = sum(len(z) for z in paths)
    constant = -len(paths) * math.log(3) - 0.5 * rows * math.log(2 * math.pi)
    reference = event_score(*probabilities(m), risk, end, transitions) + constant
    actual = m.supervised_log_likelihood(xs, paths, reduce=True)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-12)
    parameters = [p for p in m.parameters() if p.requires_grad]
    left = torch.autograd.grad(actual, parameters, allow_unused=True)
    right = torch.autograd.grad(reference, parameters, allow_unused=True)
    for p, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    summary = m.fit_supervised(xs, paths, steps=250, lr=0.08)
    q, transition, tail_probability = probabilities(m)
    hazard_mle = end / risk
    if not tail:
        hazard_mle[:, -1] = 1
    q_mle = np.zeros((3, D))
    for k in range(3):
        survival = 1.0
        for a in range(D - 1):
            q_mle[k, a] = survival * hazard_mle[k, a]
            survival *= 1 - hazard_mle[k, a]
        q_mle[k, -1] = survival
    transition_mle = transitions / transitions.sum(-1, keepdims=True)
    torch.testing.assert_close(q, torch.tensor(q_mle), atol=0.003, rtol=0)
    torch.testing.assert_close(transition, torch.tensor(transition_mle), atol=0.003, rtol=0)
    if tail:
        torch.testing.assert_close(
            tail_probability, torch.tensor(hazard_mle[:, -1]), atol=0.003, rtol=0
        )
    reference_final = event_score(*probabilities(m), risk, end, transitions) + constant
    torch.testing.assert_close(
        m.supervised_log_likelihood(xs, paths, reduce=True), reference_final, atol=1e-10, rtol=1e-12
    )
    optimum = (
        event_score(
            torch.tensor(q_mle),
            torch.tensor(transition_mle),
            torch.tensor(hazard_mle[:, -1]) if tail else None,
            risk,
            end,
            transitions,
        )
        + constant
    )
    assert float((optimum - reference_final).detach()) / rows < 1e-5
    assert summary["updates"] == 250 and summary["rows"] == rows
    assert summary["final_nll_per_row"] < summary["initial_nll_per_row"]
    record_property("duration_max_mle_error", float((q - torch.tensor(q_mle)).abs().max().detach()))
    record_property(
        "transition_max_mle_error",
        float((transition - torch.tensor(transition_mle)).abs().max().detach()),
    )
    record_property("nll_gap_per_row", float((optimum - reference_final).detach()) / rows)
    record_property("observed_edges", int(risk.sum()))


@pytest.mark.parametrize("D,tail", [(1, False), (2, False), (1, True), (2, True)])
@pytest.mark.parametrize("first_available", [False, True])
def test_joint_supervision_raw_neural_gradients_and_outgoing_timing(D, tail, first_available):
    m, x, c = fixture(D, tail, first_available)
    labels = torch.tensor([0, 0, 1, 2])
    reference, _ = path_weight(tuple(labels.tolist()), *parts(m, x, c), hmm=D == 1 and not tail)
    expected = reference.log()
    actual = m.supervised_log_likelihood(x, labels, context=c).sum()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    parameters = [p for p in m.parameters() if p.requires_grad] + [c.emission, c.boundary]
    left = torch.autograd.grad(actual, parameters, allow_unused=True)
    right = torch.autograd.grad(expected, parameters, allow_unused=True)
    for p, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    assert torch.equal(left[-2][~c.available], torch.zeros_like(left[-2][~c.available]))
    assert torch.equal(left[-1][~c.available], torch.zeros_like(left[-1][~c.available]))
    assert torch.equal(left[-1][-1], torch.zeros_like(left[-1][-1]))
