"""Independent expected counts, terminal survival and EM mathematical checks."""

import itertools

import numpy as np
import pytest

from scripts.compare_unlabelled_duration_em import (
    emission_scores,
    expectation,
    fit_em,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import sequences, true_law


def test_expectation_matches_enumerated_full_durations_including_terminal_censor():
    p = np.array([[0.3, 0.7], [0.8, 0.2]])
    transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    initial = np.array([0.4, 0.6])
    emission = np.array([[0.8, 0.2], [0.3, 0.9], [0.7, 0.4]])
    paths = []

    # Enumerate complete latent episode lengths, including terminal unobserved rows.
    def expand(t, state, weight, counts):
        for length in (1, 2):
            mass = weight * p[state, length - 1] * emission[t : min(t + length, 3), state].prod()
            c = counts.copy()
            c[state, length - 1] += 1
            if t + length >= 3:
                paths.append((mass, c))
            else:
                expand(t + length, 1 - state, mass, c)

    for k in (0, 1):
        expand(0, k, initial[k], np.zeros_like(p))
    evidence = sum(m for m, _ in paths)
    expected = sum(m * c for m, c in paths) / evidence
    ll, counts, complete, censor = expectation(np.log(emission)[None], p, transition, initial)
    assert ll[0] == pytest.approx(np.log(evidence), abs=1e-12)
    np.testing.assert_allclose(counts, expected, rtol=0, atol=1e-12)
    assert censor.sum() == pytest.approx(1)
    assert counts.sum() == pytest.approx(complete.sum() + 1)


def test_terminal_one_row_has_no_completed_event_and_zero_duration_gradient():
    p = np.array([[0.2, 0.8], [0.6, 0.4]])
    scores = np.log(np.array([[[0.8, 0.2]]]))
    _, counts, complete, _ = expectation(scores, p, np.eye(2), np.array([0.5, 0.5]))
    assert complete.sum() == 0
    np.testing.assert_allclose(counts - p * counts.sum(1, keepdims=True), 0, atol=1e-12)


def test_posterior_all_logit_score_matches_production_and_segment_likelihood():
    x, _ = sequences(123, count=1, steps=300, sigma=2.0)
    support = true_law() > 0
    p = support / support.sum(1, keepdims=True)
    assert posterior_gradient_check(x, p)["maximum_all_logit_gradient_error"] < 1e-7


def test_unlabelled_em_is_monotonic_and_score_matches_finite_differences():
    x, _ = sequences(124, count=2, steps=100, sigma=2.0)
    support = true_law() > 0
    p = support / support.sum(1, keepdims=True)
    _, counts, _, _ = expectation(
        emission_scores(x),
        p,
        np.array([[0, 0.4, 0.3, 0.3], [0.3, 0, 0.4, 0.3], [0.4, 0.3, 0, 0.3], [0.3, 0.3, 0.4, 0]]),
        np.full(4, 0.25),
    )
    # Full production transition is verified separately; this checks arbitrary chains.
    transition = np.array(
        [[0, 0.4, 0.3, 0.3], [0.3, 0, 0.4, 0.3], [0.4, 0.3, 0, 0.3], [0.3, 0.3, 0.4, 0]]
    )
    score = counts - p * counts.sum(1, keepdims=True)
    for k, d in itertools.product(range(4), [60, 90]):
        eps = 1e-4
        values = []
        for sign in (1, -1):
            q = p.copy()
            q[k, d] *= np.exp(sign * eps)
            q[k] /= q[k].sum()
            values.append(expectation(emission_scores(x), q, transition, np.full(4, 0.25))[0].sum())
        assert score[k, d] == pytest.approx((values[0] - values[1]) / (2 * eps), abs=1e-7)
    fitted, history = fit_em(x, p, iterations=3)
    assert min(np.diff([r["ll"] for r in history])) >= -1e-6
    assert history[-1]["ll"] > history[0]["ll"]
    np.testing.assert_allclose(fitted.sum(1), 1, atol=1e-12)
