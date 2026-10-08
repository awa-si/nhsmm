from __future__ import annotations

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


def _model(*, emission_type: str = 'gaussian', mode: str = 'kmeans_cluster_variance') -> NHSMM:
    cfg = ModelConfig(n_states=3, n_features=2, max_duration=5, causal=True,
                      seed=11, emission_type=emission_type, emission_init_mode=mode)
    model = NHSMM(cfg, device='cpu')
    model.initialize_distributions()
    return model


def _observations() -> torch.Tensor:
    torch.manual_seed(19)
    return torch.cat([
        torch.tensor([-8., 0.]) + torch.randn(40, 2) * torch.tensor([.2, .6]),
        torch.tensor([0., 8.]) + torch.randn(40, 2) * torch.tensor([.5, .3]),
        torch.tensor([8., 0.]) + torch.randn(40, 2) * torch.tensor([.9, .4]),
    ])


def test_cluster_variance_initialization_matches_cluster_statistics_and_preserves_parameters():
    model = _model()
    x = _observations()
    model._initialize_run_state(0, observations=x, emission_init_mode='kmeans_cluster_variance')
    emission = model.dist.emission
    before = (id(emission.mu), id(emission.log_var))
    torch.manual_seed(11)
    model._initialize_emission_from_observations(x, cluster_variance=True)
    assert (id(emission.mu), id(emission.log_var)) == before
    labels = torch.cdist(x, emission.mu).argmin(-1)
    expected = torch.stack([x[labels == k].var(0, unbiased=False) for k in range(3)])
    torch.testing.assert_close(emission.log_var.exp(), expected.clamp(.05, 5), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(emission.log_var).all()


def test_kmeans_legacy_unit_raw_variance_is_unchanged():
    model = _model(mode='kmeans')
    model._initialize_run_state(0, observations=_observations(), emission_init_mode='kmeans')
    torch.testing.assert_close(model.dist.emission.log_var, torch.zeros_like(model.dist.emission.log_var))


def test_cluster_variance_handles_single_member_and_zero_variance():
    model = _model()
    x = torch.zeros(12, 2)
    model._initialize_run_state(0, observations=x, emission_init_mode='kmeans_cluster_variance')
    torch.testing.assert_close(model.dist.emission.log_var.exp(), torch.full((3, 2), .05))
    assert torch.isfinite(model.dist.emission.log_var).all()


def test_cluster_variance_rejects_studentt():
    model = _model(emission_type='studentt')
    with pytest.raises(ValueError, match='requires gaussian'):
        model._initialize_run_state(0, observations=_observations(), emission_init_mode='kmeans_cluster_variance')


def test_cluster_variance_mode_validates_config():
    cfg = ModelConfig(n_states=3, n_features=2, emission_init_mode='kmeans_cluster_variance')
    assert cfg.emission_init_mode == 'kmeans_cluster_variance'
    assert ModelConfig.from_dict(cfg.to_dict()).emission_init_mode == "kmeans_cluster_variance"
