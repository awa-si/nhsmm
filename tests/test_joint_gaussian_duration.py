import numpy as np
import pytest
import torch

from scripts.block_mass_smoothing import marginal_objective
from scripts.compare_unlabelled_duration_em import expectation
from scripts.fixed_bin_posterior_score import density, score_gradient
from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.joint_gaussian_duration import (
    chunk_statistics,
    emission_gradients,
    initialize_from_observations,
    joint_mstep,
    scores,
    statistics,
)
from scripts.verify_isolated_trading_hsmm import make_model, sequences


def problem():
    torch.set_num_threads(1)
    x, _ = sequences(401925, count=3, steps=60, sigma=2.0)
    z = torch.zeros(4, 32, dtype=torch.float64)
    p = density(z).numpy()
    means = np.array([-5.5, -1.8, 2.5, 5.2])
    variances = np.array([3.0, 4.2, 5.0, 2.0])
    return x, p, means, variances


def test_independent_duration_statistics_and_chunk_equivalence():
    x, p, means, variances = problem()
    s = statistics(x, p, means, variances, chunk=2)
    ll, counts, _, _ = expectation(
        scores(x, means, variances), p, BASE_TRANSITION, np.full(4, 0.25)
    )
    np.testing.assert_allclose(s["ll"], ll.sum(), atol=1e-8, rtol=0)
    np.testing.assert_allclose(s["duration_counts"], counts, atol=1e-10, rtol=0)
    full = statistics(x, p, means, variances, chunk=3)
    for key in s:
        np.testing.assert_allclose(s[key], full[key], atol=1e-9, rtol=1e-12)
    assert s["occupancy"].sum() == pytest.approx(x.shape[0] * x.shape[1], abs=1e-8)


def test_public_joint_likelihood_and_all_parameter_gradients():
    x, p, means, variances = problem()
    model = make_model(p, sigma=2.0)
    raw = variances + np.log(-np.expm1(-variances))
    with torch.no_grad():
        model.dist.emission.mu.copy_(torch.tensor(means[:, None]))
        model.dist.emission.log_var.copy_(torch.tensor(raw[:, None]))
    model.dist.emission.mu.requires_grad_(True)
    model.dist.emission.log_var.requires_grad_(True)
    z = torch.nn.Parameter(torch.zeros(4, 32, dtype=torch.float64))
    pmf = density(z)
    del model.dist.duration.logits
    model.dist.duration.logits = pmf.log()
    ll = model.log_likelihood(torch.tensor(x), reduce=True)
    (ll + 1e-6 * pmf.log().sum()).backward()
    s = statistics(x, p, means, variances, chunk=2)
    assert float(ll.detach()) == pytest.approx(s["ll"], abs=1e-7)
    gm, gv = emission_gradients(s, means, variances)
    np.testing.assert_allclose(model.dist.emission.mu.grad.numpy()[:, 0], gm, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        model.dist.emission.log_var.grad.numpy()[:, 0], gv / (1 + np.exp(-raw)), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(z.grad, score_gradient(z, s["duration_counts"]), atol=1e-7, rtol=0)
    from nhsmm.filtering import filter_model_sequence

    with torch.no_grad():
        trace = filter_model_sequence(model, torch.tensor(x))
    reference = chunk_statistics(x, p, means, variances, return_filter=True)
    np.testing.assert_allclose(
        trace.state_posterior.numpy(),
        reference["filtered_state_probabilities"],
        atol=1e-8,
        rtol=0,
    )


def test_actual_joint_em_updates_monotonic_and_inputs_unchanged():
    x, p, _, _ = problem()
    original = x.copy()
    means, variances = initialize_from_observations(x)
    previous = -np.inf
    for _ in range(4):
        s = statistics(x, p, means, variances, chunk=2)
        objective = marginal_objective(s["ll"], p, 0.0)
        assert objective >= previous - 1e-6
        previous = objective
        p, means, variances, certificate = joint_mstep(s)
        assert certificate["maximum_normalized_anchored_gradient"] <= 1e-8
        assert (p > 0).all() and (variances >= 1e-6).all()
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("kind", ["variance", "pmf", "data", "chunk"])
def test_invalid_statistics_rejected(kind):
    x, p, means, variances = problem()
    chunk = 2
    if kind == "variance":
        variances[0] = 0
    elif kind == "pmf":
        p[0, 0] = -1
    elif kind == "data":
        x[0, 0, 0] = np.nan
    else:
        chunk = 0
    with pytest.raises(ValueError):
        statistics(x, p, means, variances, chunk=chunk)
