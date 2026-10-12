"""Causal edge-emission density and filtering checks for observation controls."""

import itertools
import math

import numpy as np
import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence

from scripts.diagnose_emission_dependence import (
    MEANS,
    edge_log_density,
    fit_emissions,
    filter_observations,
    gaussian_log_density,
    observations,
)


def test_stationary_residual_marginals_covariance_and_AR_are_preserved():
    labels = np.zeros((1, 50000), dtype=int)
    for rho in (0.0, 0.8):
        for correlation in (0.0, 0.75):
            x = observations(labels, 6, rho, correlation)
            u = x[0] - MEANS[0]
            np.testing.assert_allclose(u.var(0), 1.0, atol=0.05)
            assert np.corrcoef(u.T)[0, 1] == pytest.approx(correlation, abs=0.025)
            assert np.corrcoef(u[:-1, 0], u[1:, 0])[0, 1] == pytest.approx(rho, abs=0.025)


def test_generation_preserves_all_sequence_prefixes_and_extra_sequences():
    labels = np.tile(np.arange(30) % 2, (3, 1))
    full = observations(labels, 4, 0.8, 0.75)
    np.testing.assert_array_equal(full[:, :10], observations(labels[:, :10], 4, 0.8, 0.75))
    np.testing.assert_array_equal(full[:2], observations(labels[:2], 4, 0.8, 0.75))


def test_edge_density_matches_independent_multivariate_normal_and_is_past_only():
    covariance = np.array([[1.0, 0.75], [0.75, 1.0]])
    current = np.array([[0.2, -0.3]])
    previous = np.array([[-0.4, 0.6]])
    actual = edge_log_density(current, previous, MEANS, covariance, 0.8)
    for i, j in itertools.product(range(2), repeat=2):
        mean = MEANS[j] + 0.8 * (previous[0] - MEANS[i])
        dist = torch.distributions.MultivariateNormal(
            torch.tensor(mean), torch.tensor(covariance * 0.36)
        )
        assert actual[0, i, j] == pytest.approx(
            float(dist.log_prob(torch.tensor(current[0]))), abs=1e-12
        )
    # At rho=0 all previous-state rows collapse to the current-state IID law.
    zero = edge_log_density(current, previous, MEANS, covariance, 0.0)
    direct = gaussian_log_density(current[:, None], MEANS, covariance)
    np.testing.assert_allclose(zero[:, 0], direct)
    np.testing.assert_allclose(zero[:, 1], direct)


def test_filter_matches_independent_complete_state_paths_and_prefixes():
    x = np.array([[[0.2, -0.3], [0.4, -0.1], [-0.5, 0.1]]])
    covariance = np.array([[1.0, 0.6], [0.6, 1.0]])
    pmf = np.ones((2, 1))
    hazard = np.array([0.2, 0.4])
    transition = np.array([[0.1, 0.9], [0.8, 0.2]])
    effective = (1 - hazard)[:, None] * np.eye(2) + hazard[:, None] * transition
    posterior, increments = filter_observations(x, MEANS, covariance, 0.8, pmf, hazard, transition)
    for t in range(3):
        mass = np.zeros(2)
        for path in itertools.product(range(2), repeat=t + 1):
            p = 0.5 * np.exp(gaussian_log_density(x[0, 0], MEANS[path[0]], covariance))
            for s in range(1, len(path)):
                mean = MEANS[path[s]] + 0.8 * (x[0, s - 1] - MEANS[path[s - 1]])
                p *= effective[path[s - 1], path[s]] * np.exp(
                    gaussian_log_density(x[0, s], mean, covariance * 0.36)
                )
            mass[path[-1]] += p
        np.testing.assert_allclose(posterior[0, t], mass / mass.sum(), atol=1e-12)
        assert increments[0, : t + 1].sum() == pytest.approx(np.log(mass.sum()), abs=1e-12)
    mutated = x.copy()
    mutated[:, 2] += 100
    np.testing.assert_allclose(
        filter_observations(mutated, MEANS, covariance, 0.8, pmf, hazard, transition)[0][:, :2],
        posterior[:, :2],
    )
    np.testing.assert_allclose(
        filter_observations(x[:, :2], MEANS, covariance, 0.8, pmf, hazard, transition)[0],
        posterior[:, :2],
    )


def test_training_nuisance_fit_does_not_consume_held_out_values():
    labels = np.tile(np.arange(100) % 2, (3, 1))
    train = observations(labels, 5, 0.8, 0.75)
    before = fit_emissions(train, labels)
    test = observations(labels, 6, 0.8, 0.75)
    test[:] = 999
    after = fit_emissions(train, labels)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("tail", [False, True])
def test_IID_limit_matches_public_HSMM_for_nonuniform_explicit_durations(tail):
    pmf = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    hazard = np.array([0.2, 0.4]) if tail else np.ones(2)
    transition = np.array([[0.1, 0.9], [0.8, 0.2]])
    x = np.array([[[0.2, -0.3], [0.4, -0.1], [-0.5, 0.1], [0.8, 0.6]]])
    # Public NHSMM input canonicalization uses Float32 even for double parameters.
    # Compare identical observation values, keeping strict Float64 recursion checks.
    x = x.astype(np.float32).astype(np.float64)
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=2,
            max_duration=3,
            causal=True,
            duration_tail=tail,
            use_context_encoder=False,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.double().eval()
    with torch.no_grad():
        model.dist.initial.logits.fill_(0)
        model.duration_logits_bias.fill_(1)
        model.dist.duration.logits.copy_(torch.tensor(np.log(pmf)))
        if tail:
            model.dist.duration.tail_end_probability.copy_(torch.tensor(hazard))
        model.dist.transition.logits.copy_(
            torch.tensor(np.log(transition))[:, None].expand(2, 3, 2)
        )
        model.dist.emission.mu.copy_(torch.tensor(MEANS))
        model.dist.emission.log_var.fill_(math.log(math.expm1(1)))
        tensor = torch.tensor(x)
        trace = filter_model_sequence(model, tensor)
        ll = float(model.log_likelihood(tensor, reduce=True))
    posterior, increments = filter_observations(x, MEANS, np.eye(2), 0.0, pmf, hazard, transition)
    np.testing.assert_allclose(posterior, trace.state_posterior.numpy(), atol=1e-12)
    assert increments.sum() == pytest.approx(ll, abs=1e-12)
