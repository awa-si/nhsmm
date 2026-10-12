"""Censored duration MLE and independent finite-support HSMM checks."""

import numpy as np
import pytest
import torch

from scripts.verify_isolated_trading_hsmm import (
    censored_counts,
    check_recursion,
    count_ll,
    fit_duration,
    labelled_em,
    make_model,
    model_law,
    segment_reference,
    sequences,
    true_law,
)


def test_labelled_em_uses_survival_and_matches_hand_mle_at_terminal_support():
    complete = np.array([[4.0, 2.0, 0.0]])
    censored = np.array([[0.0, 0.0, 4.0]])
    fitted, history = labelled_em(complete, censored, np.ones((1, 3), dtype=bool), 5)
    np.testing.assert_allclose(fitted, [[0.4, 0.2, 0.4]], rtol=0, atol=1e-12)
    assert history[-1] == pytest.approx(8 * np.log(0.4) + 2 * np.log(0.2))
    p = np.array([[0.3, 0.2, 0.5]])
    assert count_ll(p, np.zeros_like(p), np.array([[0.0, 1.0, 0.0]])) == pytest.approx(np.log(0.7))


def test_segment_reference_matches_independent_enumerated_hidden_age_paths():
    pmf = np.array([[0.4, 0.6], [0.7, 0.3]])
    transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    initial = np.array([0.3, 0.7])
    emission = np.array([[0.8, 0.2], [0.5, 0.4], [0.1, 0.9]])
    frontier = {(0, 1): initial[0], (1, 1): initial[1]}
    posterior = []
    for t in range(3):
        if t:
            next_frontier = {}
            for (k, age), mass in frontier.items():
                h = pmf[k, age - 1] / pmf[k, age - 1 :].sum()
                for j in range(2):
                    key = (j, 1)
                    next_frontier[key] = next_frontier.get(key, 0) + mass * h * transition[k, j]
                if age < 2:
                    key = (k, age + 1)
                    next_frontier[key] = next_frontier.get(key, 0) + mass * (1 - h)
            frontier = next_frontier
        frontier = {key: mass * emission[t, key[0]] for key, mass in frontier.items()}
        state = np.array(
            [sum(mass for (k, a), mass in frontier.items() if k == j) for j in range(2)]
        )
        posterior.append(state / state.sum())
    ll, actual = segment_reference(np.log(emission), pmf, transition, initial)
    assert ll == pytest.approx(np.log(sum(frontier.values())), abs=1e-12)
    np.testing.assert_allclose(np.exp(actual), posterior, rtol=0, atol=1e-12)


def test_oracle_support_no_reset_and_public_recursion_are_exactly_matched():
    model = make_model(true_law())
    p = model_law(model).detach().numpy()
    np.testing.assert_allclose(p, true_law(), rtol=0, atol=1e-14)
    with torch.inference_mode():
        transition = model.dist.transition.log_matrix(T=1).exp().numpy()[0, 0]
    assert all(transition[k, :, k].sum() == 0 for k in range(4))
    x, labels = sequences(17, count=2, steps=300)
    result = check_recursion(model, x, labels)
    assert result["accuracy"] >= 0.99


def test_duration_gradient_updates_reduce_censored_objective_with_frozen_nuisance():
    pmf = true_law()
    complete, censored = censored_counts(18, count=64)
    initial = make_model(pmf, uniform=True)
    before = count_ll(model_law(initial).detach().numpy(), complete, censored)
    fitted, history = fit_duration(pmf, complete, censored, iterations=30)
    after = count_ll(model_law(fitted).detach().numpy(), complete, censored)
    assert after > before
    assert history[-1]["completed_updates"] == 30
    assert torch.isfinite(fitted.dist.duration.logits.grad).all()


def test_public_marginal_recursion_with_overlapping_emissions_has_nontrivial_posteriors():
    model = make_model(true_law(), sigma=2.0)
    x, labels = sequences(19, count=2, steps=300, sigma=2.0)
    result = check_recursion(model, x, labels, sigma=2.0)
    assert result["uncertain_rows_max_probability_below_0_99"] > 0
