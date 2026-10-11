"""Independent density and leakage checks for the feature diagnostic."""

import numpy as np
import pytest

from scripts.diagnose_trading_feature_redundancy import fit_whitening


def test_training_whitening_covariance_and_pointwise_prefix():
    rng = np.random.default_rng(17)
    source = rng.normal(size=(4, 50, 3))
    transform = np.array([[2.0, 0.3, -0.4], [0.0, 1.0, 0.2], [0.0, 0.0, 0.5]])
    train = source @ transform + np.array([5.0, 2.0, -4.0])
    fitted = fit_whitening(train, eigenvalue_floor=1e-8)
    actual = fitted.apply(train).reshape(-1, 3)
    np.testing.assert_allclose(actual.mean(0), np.zeros(3), atol=2e-7)
    np.testing.assert_allclose(np.cov(actual, rowvar=False, bias=True), np.eye(3), atol=2e-7)
    future = rng.normal(size=(1, 20, 3)) * 100
    before = fitted.matrix.copy()
    np.testing.assert_array_equal(fitted.apply(future)[:, :7], fitted.apply(future[:, :7]))
    np.testing.assert_array_equal(fitted.matrix, before)


def test_whitening_jacobian_matches_independent_gaussian_density():
    train = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])[None]
    fitted = fit_whitening(train)
    x = np.array([0.7, -1.2])
    z = (x - fitted.mean) @ fitted.matrix
    unit_lp = -0.5 * (2 * np.log(2 * np.pi) + z @ z)
    cov = np.cov(train.reshape(-1, 2), rowvar=False, bias=True)
    delta = x - fitted.mean
    expected = -0.5 * (
        2 * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + delta @ np.linalg.solve(cov, delta)
    )
    assert unit_lp + fitted.log_abs_det == pytest.approx(expected, abs=1e-12)
    assert fitted.log_abs_det == pytest.approx(np.linalg.slogdet(fitted.matrix)[1])


def test_whitening_regularization_retains_invertible_map():
    fitted = fit_whitening(np.ones((2, 5, 3)))
    assert np.isfinite(fitted.matrix).all()
    assert np.linalg.matrix_rank(fitted.matrix) == 3
    assert np.isfinite(fitted.log_abs_det)
    with pytest.raises(ValueError):
        fit_whitening(np.ones((2, 5, 3)), eigenvalue_floor=0)
