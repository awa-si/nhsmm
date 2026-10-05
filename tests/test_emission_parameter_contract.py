from __future__ import annotations

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.distributions.default import Emission


def _effective_gaussian_variance(emission: Emission) -> torch.Tensor:
    return torch.nn.functional.softplus(emission.log_var).clamp_min(emission.min_covar)


def _effective_studentt_scale(emission: Emission) -> torch.Tensor:
    return torch.nn.functional.softplus(emission.scale_param).clamp_min(emission.min_covar)


def test_gaussian_initialize_preserves_legacy_effective_variance() -> None:
    emission = Emission(
        n_states=3,
        n_features=4,
        activation="tanh",
        emission_type="gaussian",
        context_dim=None,
    )
    emission.initialize(mode="spread", jitter=0.0)

    torch.testing.assert_close(
        _effective_gaussian_variance(emission),
        torch.full((3, 4), torch.nn.functional.softplus(torch.tensor(0.0)).item()),
        atol=1e-6,
        rtol=1e-6,
    )


def test_studentt_initialize_preserves_legacy_effective_scale() -> None:
    emission = Emission(
        n_states=3,
        n_features=4,
        activation="tanh",
        emission_type="studentt",
        context_dim=None,
    )
    emission.initialize(mode="spread", jitter=0.0)

    torch.testing.assert_close(
        _effective_studentt_scale(emission),
        torch.full((3, 4), torch.nn.functional.softplus(torch.tensor(1.0)).item()),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("emission_type", ["gaussian", "studentt"])
def test_emission_initialize_preserves_parameter_identity(emission_type: str) -> None:
    emission = Emission(
        n_states=2,
        n_features=3,
        activation="tanh",
        emission_type=emission_type,
        context_dim=None,
    )

    if emission_type == "gaussian":
        before = (id(emission.mu), id(emission.log_var))
    else:
        before = (id(emission.loc), id(emission.scale_param), id(emission.dof))

    emission.initialize(mode="spread", jitter=0.0)

    if emission_type == "gaussian":
        after = (id(emission.mu), id(emission.log_var))
    else:
        after = (id(emission.loc), id(emission.scale_param), id(emission.dof))

    assert after == before


def test_kmeans_initialize_preserves_emission_parameter_identity_and_unit_variance() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=5,
        causal=True,
        seed=7,
        emission_init_mode="kmeans",
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    emission = model.dist.emission
    observations = torch.cat(
        [
            torch.tensor([-3.0, 0.0]) + 0.05 * torch.randn(30, 2),
            torch.tensor([0.0, 3.0]) + 0.05 * torch.randn(30, 2),
            torch.tensor([3.0, 0.0]) + 0.05 * torch.randn(30, 2),
        ],
        dim=0,
    )

    model._initialize_run_state(
        0,
        observations=observations,
        encoder_state=model._clone_state_dict(model.encoder),
        emission_init_mode="kmeans",
    )

    emission = model.dist.emission
    after = (id(emission.mu), id(emission.log_var))

    # fresh() legitimately creates a new DistributionSet; within the fresh emission,
    # K-Means initialization itself must not replace its registered Parameters.
    fresh_before = after
    model._initialize_emission_from_observations(observations)
    fresh_after = (id(model.dist.emission.mu), id(model.dist.emission.log_var))

    assert fresh_after == fresh_before
    torch.testing.assert_close(
        _effective_gaussian_variance(model.dist.emission),
        torch.full_like(
            model.dist.emission.log_var,
            torch.nn.functional.softplus(torch.tensor(0.0)).item(),
        ),
        atol=1e-6,
        rtol=1e-6,
    )
