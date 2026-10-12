"""Causal observable heads, fixed training calibration, and unfloored state logs."""

import numpy as np
import pytest
import torch

from scripts.diagnose_trading_information_baselines import (
    LAGS,
    WARMUP,
    fit_head,
    observable_design,
    raw_state_scores,
    score_head,
)


def test_observable_lags_use_current_prefix_and_never_cross_windows():
    x = np.arange(200, dtype=float).reshape(2, 100, 1)
    c = -x
    design = observable_design(x, c)
    for index, lag in enumerate(LAGS):
        np.testing.assert_array_equal(
            design[:, 1 + index * 2], x[:, WARMUP - lag : 99 - lag].reshape(-1)
        )
    changed = x.copy()
    changed[:, 80:] = 1e6
    before = design.reshape(2, 35, -1)[:, :16]
    after = observable_design(changed, c).reshape(2, 35, -1)[:, :16]
    np.testing.assert_array_equal(before, after)
    changed = x.copy()
    changed[:, -1] = 1e6
    np.testing.assert_array_equal(design, observable_design(changed, c))
    with pytest.raises(ValueError, match="causal"):
        observable_design(x, c, lags=(-1,))


@pytest.mark.parametrize("hetero", [False, True])
def test_frozen_training_head_and_raw_units_independent_normal(hetero):
    rng = np.random.default_rng(24)
    train = rng.normal(0, 0.002, (4, 100))
    train_design = observable_design(train[..., None], np.ones((4, 100, 1)))
    head = fit_head(train_design, train, heteroscedastic=hetero)
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in head.items()}
    test = rng.normal(0, 0.002, (2, 100))
    design = observable_design(test[..., None], np.ones((2, 100, 1)))
    result = score_head(design, test, head)
    variance = np.maximum(head["smear"] * np.exp(np.clip(design @ head["gamma"], -12, 12)), 1e-6)
    distribution = torch.distributions.Normal(
        torch.tensor(head["mean"] + head["scale"] * (design @ head["beta"])),
        torch.tensor(head["scale"] * np.sqrt(variance)),
    )
    target = torch.tensor(test[:, WARMUP + 1 :].reshape(-1))
    assert result["pairs"] == 70
    assert result["next_return_log_score"] == pytest.approx(
        float(distribution.log_prob(target).mean()), abs=1e-12
    )
    test[:] = 1e6
    score_head(design, test, head)
    for key in before:
        np.testing.assert_array_equal(head[key], before[key])


def test_raw_log_probability_underflow_and_permutation_are_scored_without_floor():
    logp = np.full((1, 100, 4), -1000, dtype=np.float32)
    logp[..., 0] = 0
    labels = np.zeros((1, 100), dtype=int)
    result = raw_state_scores(logp, labels, (2, 0, 3, 1))
    assert result["state_log_score"] == -1000
    assert result["minimum_true_state_log_probability"] == -1000
    assert result["float32_exp_zero_true_probabilities"] == 36
    assert result["brier"] == 2
    assert result["accuracy"] == 0
    logp[..., 1] = -np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        raw_state_scores(logp, labels, (2, 0, 3, 1))


def test_constant_posterior_addition_reproduces_direct_head_predictions():
    rng = np.random.default_rng(11)
    returns = rng.normal(0, 0.002, (4, 100))
    x, c = returns[..., None], np.zeros((4, 100, 1))
    p = np.full((4, 100, 4), 0.25)
    direct = observable_design(x, c)
    augmented = observable_design(x, c, p)
    a = fit_head(direct, returns, heteroscedastic=True)
    b = fit_head(augmented, returns, heteroscedastic=True)
    assert score_head(direct, returns, a)["next_return_log_score"] == pytest.approx(
        score_head(augmented, returns, b)["next_return_log_score"], abs=1e-12
    )
