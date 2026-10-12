"""Observation-only duration objective and independent derivative checks."""

import numpy as np
import pytest
import torch

from scripts.verify_isolated_trading_hsmm import make_model, model_law, sequences, true_law
from scripts.verify_unlabelled_trading_hsmm import (
    SIGMA,
    exposure,
    fit_observations,
    gradient_check,
    independent_ll,
    marginal_ll,
)


def test_independent_marginal_objective_and_gradients_with_uncertain_states():
    x, _ = sequences(44, count=1, steps=300, sigma=SIGMA)
    model = make_model(true_law(), uniform=True, sigma=SIGMA)
    assert marginal_ll(model, x) == pytest.approx(
        independent_ll(x, model_law(model).detach().numpy()), abs=1e-7
    )
    assert gradient_check(model, x)["maximum_absolute_error"] < 1e-5


def test_observation_only_fit_moves_duration_and_preserves_nuisance():
    x, labels = sequences(45, count=2, steps=300, sigma=SIGMA)
    initial = make_model(true_law(), uniform=True, sigma=SIGMA)
    before = marginal_ll(initial, x)
    model, report = fit_observations(x, iterations=3, seed=45)
    assert marginal_ll(model, x) > before
    assert report["actual_updates"] == 3
    assert report["frozen_nuisance_unchanged"]
    assert torch.isfinite(model.dist.duration.logits.grad).all()
    # Evaluation labels do not share observation storage and cannot alter its objective.
    labels[:] = (labels + 1) % 4
    assert marginal_ll(initial, x) == before


def test_exposure_treats_every_terminal_segment_as_right_censored():
    labels = np.array([[0] * 42 + [1] * 30 + [2] * 18, [3] * 90])
    result = exposure(labels)
    assert result["complete_episodes_by_state"] == [1, 1, 0, 0]
    assert result["terminal_censored_episodes_by_state"] == [0, 0, 1, 1]
