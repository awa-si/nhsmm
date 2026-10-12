"""Observation-control generation contracts independent of model training."""

import numpy as np
import pytest

from scripts.diagnose_trading_observation_controls import (
    control_observations,
    gaussian_control,
    training_moments,
)
from scripts.validate_trading_regime_usefulness import FEATURES, K


def test_training_moments_match_known_diagonal_gaussian_statistics():
    labels = np.repeat(np.arange(K), 4)
    x = np.stack(
        [labels * 3.0 + np.tile([-3, -1, 1, 3], K), labels * -2.0 + np.tile([-6, -2, 2, 6], K)],
        axis=-1,
    )
    means, variances = training_moments(x, labels)
    np.testing.assert_allclose(means, np.column_stack([np.arange(K) * 3, np.arange(K) * -2]))
    np.testing.assert_allclose(variances, np.tile([5.0, 20.0], (K, 1)))


def test_held_out_feature_mutations_do_not_change_generated_controls():
    rng = np.random.default_rng(32)
    n = 400
    labels = np.arange(n) % K
    x = rng.normal(size=(n, len(FEATURES)))
    data = {feature: x[:, i].copy() for i, feature in enumerate(FEATURES)}
    data["gt_regime_id"] = labels
    train = (x[:200][None], None, labels[:200][None])
    controls, moments = control_observations(data, train)
    changed = {key: value.copy() for key, value in data.items()}
    for feature in FEATURES:
        changed[feature][200:] = 1e8
    updated, new_moments = control_observations(changed, train)
    assert moments == new_moments
    for arm in ("matched_iid", "matched_ar1", "separated_iid"):
        for feature in FEATURES:
            np.testing.assert_array_equal(controls[arm][feature], updated[arm][feature])
        np.testing.assert_array_equal(controls[arm]["gt_regime_id"], labels)


def test_gaussian_population_marginals_and_serial_correlation():
    labels = np.zeros(120000, dtype=int)
    means = np.tile([3.0, -2.0], (K, 1))
    variances = np.tile([4.0, 9.0], (K, 1))
    for rho in (0.0, 0.9):
        x = gaussian_control(labels, means, variances, seed=443, rho=rho)
        np.testing.assert_allclose(x.mean(0), means[0], atol=0.15)
        np.testing.assert_allclose(x.var(0), variances[0], rtol=0.07)
        assert np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1] == pytest.approx(rho, abs=0.02)
        assert abs(np.corrcoef(x.T)[0, 1]) < 0.05


def test_generation_is_reproducible_and_prefix_invariant():
    labels = np.arange(100) % K
    means = np.arange(K * 3).reshape(K, 3).astype(float)
    variances = np.ones_like(means)
    for rho in (0.0, 0.9):
        full = gaussian_control(labels, means, variances, seed=76, rho=rho)
        short = gaussian_control(labels[:31], means, variances, seed=76, rho=rho)
        np.testing.assert_array_equal(full[:31], short)
        np.testing.assert_array_equal(
            full, gaussian_control(labels, means, variances, seed=76, rho=rho)
        )


def test_missing_training_state_and_invalid_generator_support_fail():
    with pytest.raises(ValueError, match="two training observations"):
        training_moments(np.ones((4, 2)), np.zeros(4, dtype=int))
    means, variances = np.zeros((K, 2)), np.ones((K, 2))
    with pytest.raises(ValueError, match="abs"):
        gaussian_control(np.zeros(4, dtype=int), means, variances, rho=1.0)
    with pytest.raises(ValueError, match="state range"):
        gaussian_control(np.array([K]), means, variances)
    with pytest.raises(ValueError, match="strictly positive"):
        gaussian_control(np.zeros(4, dtype=int), means, variances * 0)


def test_missing_state_penalty_remains_json_serializable():
    import json
    from scripts.diagnose_trading_observation_controls import serializable_score

    truth = np.tile(np.arange(K), (2, 3))
    result = serializable_score(truth, np.zeros_like(truth))
    assert result["run_duration_rel_error"] == 1.0
    assert result["missing_states"]
    assert any(v is None for v in result["run_medians"].values())
    json.dumps(result, allow_nan=False)
