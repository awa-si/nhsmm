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


def test_emission_weight_probe_preserves_causal_prefixes_and_parameters():
    from nhsmm.filtering import filter_model_sequence
    from scripts.diagnose_trading_emission_weight import emission_weight, WEIGHTS

    torch.manual_seed(1729)
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=4,
            causal=True,
            duration_tail=True,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.eval()
    x = torch.randn(1, 9, 1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    original = model._build_sequence_set
    reference = filter_model_sequence(model, x).state_posterior
    for weight in WEIGHTS:
        with emission_weight(model, weight):
            full = filter_model_sequence(model, x).state_posterior
            prefix = filter_model_sequence(model, x[:, :5]).state_posterior
            torch.testing.assert_close(full[:, :5], prefix)
            torch.testing.assert_close(full.sum(-1), torch.ones(1, 9))
            if weight == 1:
                torch.testing.assert_close(full, reference)
        assert model._build_sequence_set == original
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
