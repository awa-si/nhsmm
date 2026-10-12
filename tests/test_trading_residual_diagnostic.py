"""Boundary, leakage and statistical controls for oracle residual diagnostics."""

import json

import numpy as np
import pytest

from scripts.diagnose_trading_residuals import fit_controls, lag_mask, residual_variants, summarize


def toy(seed, rho, count=2000):
    labels = np.repeat(np.arange(4), count)
    episodes = labels.copy()
    residual = np.random.default_rng(seed).normal(size=(len(labels), 2))
    for t in range(1, len(labels)):
        if episodes[t] == episodes[t - 1]:
            residual[t] = rho * residual[t - 1] + np.sqrt(1 - rho * rho) * residual[t]
    return residual + labels[:, None], labels, episodes


def test_lag_masks_do_not_join_separate_episodes_of_the_same_regime():
    episodes = np.array([0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(lag_mask(episodes, 1), [True, False, True, False, True])
    np.testing.assert_array_equal(lag_mask(episodes, 3), False)


def test_train_parameters_ignore_held_out_changes_and_split_first_row_is_omitted():
    train, y, e = toy(4, 0.8, 200)
    before = fit_controls(train, y, e)
    test, ty, te = toy(5, 0.8, 200)
    variants = residual_variants(test, ty, te, before)
    assert np.isnan(variants["var1"][0]).all()
    assert np.isnan(variants["var1"][200]).all()
    test[20:] = 999
    after = fit_controls(train, y, e)
    for key in ("mean", "scale", "state_mean"):
        np.testing.assert_array_equal(before[key], after[key])
    for a, b in zip(before["matrix"], after["matrix"]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("rho", [0.0, 0.8])
def test_IID_null_and_AR_residual_whitening_on_independent_test_rows(rho):
    train, y, e = toy(10, rho)
    test, ty, te = toy(11, rho)
    parameters = fit_controls(train, y, e)
    variants = residual_variants(test, ty, te, parameters)
    for state in range(4):
        raw = summarize(variants["mean_only"], ty, te, state, ["a", "b"])
        whitened = summarize(variants["var1"], ty, te, state, ["a", "b"])
        np.testing.assert_allclose(raw["acf"]["1"], rho, atol=0.07)
        np.testing.assert_allclose(whitened["acf"]["1"], 0.0, atol=0.08)


def test_constant_features_have_explicit_unidentified_statistics_and_valid_JSON():
    x = np.zeros((100, 2))
    y = np.zeros(100, dtype=int)
    e = y.copy()
    row = summarize(x, y, e, 0, ["a", "b"])
    assert row["acf"]["1"] == [None, None]
    assert row["excess_kurtosis"] == [None, None]
    json.dumps(row, allow_nan=False)
