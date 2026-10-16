"""Functional checks for the variance-aware Gaussian initializer."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from nhsmm import ModelConfig, NHSMM


def test_cluster_variance_initializer_matches_gaussian_density(monkeypatch):
    centers = torch.tensor([[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]])
    offsets = torch.tensor([[0.1, 1.0, 3.0], [0.2, 2.0, 4.0]])
    observations = torch.stack(
        [
            centers[0] - offsets[0],
            centers[0] + offsets[0],
            centers[1] - offsets[1],
            centers[1] + offsets[1],
        ]
    )
    cfg = ModelConfig(
        n_states=2,
        n_features=3,
        emission_init_mode="kmeans_cluster_variance",
        use_context_encoder=False,
        verbose=False,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    monkeypatch.setattr(model, "_kmeans_centers", lambda x, k: centers)
    with torch.no_grad():
        model._initialize_emission_from_observations(observations, cluster_variance=True)
    expected_var = torch.tensor([[0.05, 1.0, 5.0], [0.05, 4.0, 5.0]])
    torch.testing.assert_close(
        F.softplus(model.dist.emission.log_var), expected_var, rtol=1e-6, atol=1e-6
    )
    x = torch.tensor([[[0.0, 2.0, -1.0]]])
    expected_lp = -0.5 * (
        (x[..., None, :] - centers).square() / expected_var
        + expected_var.log()
        + math.log(2 * math.pi)
    ).sum(-1)
    torch.testing.assert_close(model.dist.emission.log_prob(x), expected_lp, rtol=1e-6, atol=1e-5)


def test_cluster_variance_singleton_uses_actual_global_variance(monkeypatch):
    observations = torch.tensor([[-0.5], [0.5], [8.0]])
    centers = torch.tensor([[0.0], [8.0]])
    cfg = ModelConfig(n_states=2, n_features=1, use_context_encoder=False, verbose=False)
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    monkeypatch.setattr(model, "_kmeans_centers", lambda x, k: centers)
    with torch.no_grad():
        model._initialize_emission_from_observations(observations, cluster_variance=True)
    expected = torch.tensor([[0.25], [5.0]])
    torch.testing.assert_close(
        F.softplus(model.dist.emission.log_var), expected, rtol=1e-6, atol=1e-6
    )
    assert np.isfinite(model.dist.emission.log_prob(observations[None]).detach().numpy()).all()
