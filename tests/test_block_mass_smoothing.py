"""Analytic derivatives, concavity, optimizer certificates and actual HSMM smoke."""

import numpy as np
import pytest
import torch

from scripts.block_mass_smoothing import count_objective, geometry, marginal_objective, smooth_mstep
from scripts.compare_adaptive_duration_em import PSEUDOCOUNT


def test_spacing_and_no_artificial_endpoint_neighbors():
    blocks, lengths, r = geometry(260, 8)
    assert len(blocks) == 33 and blocks[-1] == (256, 260) and lengths[-1] == 4
    centers = np.array([(a + b - 1) / 2 for a, b in blocks]) / 8
    np.testing.assert_allclose(r @ np.ones(33), 0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(r @ centers, 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(r[-1, -3:], [1, -7 / 3, 4 / 3], rtol=0, atol=1e-15)


@pytest.mark.parametrize("strength", [0.0, 1.7, 20.0])
def test_autograd_gradient_hessian_match_independent_formula(strength):
    _, lengths, r = geometry(14, 4)
    masses = np.array([7.0, 2.0, 13.0, 3.0])
    z_np = np.array([-0.7, 0.3, -0.2, 0.5])
    z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)

    def q(z):
        return count_objective(
            z[None, :],
            torch.tensor(masses[None, :]),
            torch.tensor(lengths, dtype=torch.float64),
            torch.tensor(r),
            strength,
        ).sum()

    actual = torch.autograd.grad(q(z), z, create_graph=True)[0].detach().numpy()
    mass = np.exp(z_np) * lengths
    mass /= mass.sum()
    total = masses.sum()
    expected = masses - total * mass - strength * r.T @ r @ z_np
    h = -total * (np.diag(mass) - np.outer(mass, mass)) - strength * r.T @ r
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        torch.autograd.functional.hessian(q, z).numpy(), h, rtol=0, atol=1e-12
    )
    assert np.linalg.eigvalsh(h).max() < 1e-12
    assert np.linalg.eigvalsh(h[:-1, :-1]).max() < 0


def test_lambda_zero_equals_augmented_analytic_block_mass_with_partial_bin():
    counts = np.arange(24, dtype=float).reshape(2, 12)
    p, e = smooth_mstep(counts, 0.0, width=5)
    expected = np.empty_like(p)
    q = counts + PSEUDOCOUNT
    for a, b in ((0, 5), (5, 10), (10, 12)):
        expected[:, a:b] = q[:, a:b].sum(1, keepdims=True) / q.sum(1, keepdims=True) / (b - a)
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-15)
    assert e["solver_iterations"] == 0 and e["maximum_normalized_anchored_gradient"] < 1e-12


def test_regularized_optimum_beats_independent_exhaustive_log_density_grid():
    counts = np.array([[0.0, 0.0, 3.0, 0.0, 12.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    p, e = smooth_mstep(counts, 1.7, width=4)
    c = np.array([(counts[0, a : a + 4] + PSEUDOCOUNT).sum() for a in (0, 4, 8)])
    z = np.log(p[0, [0, 4, 8]])
    z -= z[-1]

    def q(z):
        return (
            c @ z - c.sum() * np.log(np.exp(z).sum() * 4) - 1.7 * (z[0] - 2 * z[1] + z[2]) ** 2 / 2
        )

    best = max(
        q(np.array([a, b, 0.0])) for a in np.linspace(-8, 8, 81) for b in np.linspace(-8, 8, 81)
    )
    assert q(z) >= best - 1e-7
    assert e["maximum_normalized_anchored_gradient"] <= 1e-8
    assert np.all(p > 0) and np.allclose(p.sum(1), 1, rtol=0, atol=1e-12)


def test_uniform_zero_exposure_and_one_bin_remain_valid():
    p, e = smooth_mstep(np.zeros((4, 260)), np.log(1024))
    np.testing.assert_allclose(p, 1 / 260, rtol=0, atol=1e-15)
    assert e["maximum_normalized_anchored_gradient"] < 1e-8
    p, e = smooth_mstep(np.zeros((1, 4)), 5.0, width=8)
    np.testing.assert_allclose(p, 0.25, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "counts,strength,width",
    [
        ([[-1.0]], 1.0, 8),
        ([[np.nan]], 1.0, 8),
        ([[1.0]], -1.0, 8),
        ([[1.0]], 1.0, 0),
        ([[1.0]], 1.0, True),
    ],
)
def test_invalid_inputs_fail_closed(counts, strength, width):
    with pytest.raises(ValueError):
        smooth_mstep(counts, strength, width=width)


def test_nonconverged_solver_and_invalid_initial_state_rejected():
    counts = np.arange(260, dtype=float)[None, :]
    with pytest.raises(FloatingPointError):
        smooth_mstep(
            counts,
            np.log(1024),
            initial_pmf=np.full((1, 260), 1 / 260),
            max_iterations=1,
            max_evaluations=100,
            gradient_tolerance=1e-16,
        )
    with pytest.raises(ValueError):
        smooth_mstep(counts, 1.0, initial_pmf=np.zeros((1, 260)))


def test_actual_short_censored_em_and_public_numerical_smoke(monkeypatch):
    from scripts.compare_compact_duration_em import public_check
    from scripts.compare_unlabelled_duration_em import (
        aggregate,
        emission_scores,
        posterior_gradient_check,
    )
    from scripts.verify_isolated_trading_hsmm import sequences

    torch.set_num_threads(1)
    x, labels = sequences(721, count=2, steps=60, sigma=2.0)
    import scripts.compare_adaptive_duration_em as adaptive

    monkeypatch.setattr(adaptive, "true_law", lambda: pytest.fail("truth entered fitting"))
    p = np.full((4, 260), 1 / 260)
    scores = emission_scores(x)
    values = []
    for update in range(4):
        ll, counts = aggregate(scores, p)
        values.append(marginal_objective(ll, p, np.log(1024)))
        if update < 3:
            previous = p
            original = p.copy()
            p, e = smooth_mstep(counts, np.log(1024), initial_pmf=p)
            assert np.isfinite(p).all() and np.all(p > 0)
            assert e["maximum_normalized_anchored_gradient"] <= 1e-8
            np.testing.assert_array_equal(previous, original)
    assert min(np.diff(values)) >= -1e-6
    support = np.ones_like(p, dtype=bool)
    assert public_check(p, x, labels, support=support)["maximum_state_probability_error"] < 1e-8
    assert max(posterior_gradient_check(x[:1], p, support=support).values()) < 1e-7
