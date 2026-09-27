from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nhsmm.config import ModelConfig
from nhsmm.models.base import NHSMM
from nhsmm.validation import ContextEvidenceConfig
from nhsmm.validation.state_effect import (
    duration_context_tensor,
    emission_context_tensor,
    evaluate_state_context_replication,
)


@dataclass
class FakeModel:
    centers: np.ndarray
    base: np.ndarray
    slope: np.ndarray


def _centers(model):
    return model.centers


def _effect(model, context):
    x = float(np.asarray(context).reshape(-1)[0])
    return model.base + x * model.slope


def _policy(min_amp=0.05):
    return ContextEvidenceConfig(
        replication_corr_min=0.55,
        min_replica_amplitude=min_amp,
    )


def test_generic_state_effect_is_permutation_invariant():
    centers = np.array([[-2.0], [0.0], [2.0]])
    base = np.zeros((3, 4))
    slope = np.array(
        [
            [1.0, -1.0, 0.5, -0.5],
            [0.2, -0.2, 0.4, -0.4],
            [-0.7, 0.7, 0.3, -0.3],
        ]
    )
    a = FakeModel(centers, base, slope)
    p = np.array([2, 0, 1])
    b = FakeModel(centers[p], base[p], slope[p])

    result = evaluate_state_context_replication(
        a,
        b,
        [-1.0, 0.0, 1.0],
        effect_tensor=_effect,
        config=_policy(),
        centers=_centers,
    )

    assert result.selected
    assert result.replication_corr > 0.999


def test_generic_state_effect_rejects_null_effect():
    centers = np.array([[-1.0], [0.0], [1.0]])
    a = FakeModel(centers, np.ones((3, 2)), np.zeros((3, 2)))
    b = FakeModel(centers.copy(), np.ones((3, 2)), np.zeros((3, 2)))

    result = evaluate_state_context_replication(
        a,
        b,
        [-1.0, 1.0],
        effect_tensor=_effect,
        config=_policy(),
        centers=_centers,
    )

    assert not result.selected
    assert result.min_replica_amplitude == 0.0


def _model(seed=1):
    cfg = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=4,
        context_dim=2,
        hidden_dim=2,
        seed=seed,
        causal=True,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    return model


def test_duration_adapter_returns_normalized_state_duration_tensor_and_preserves_mode():
    model = _model()
    model.train()
    value = duration_context_tensor(model, [0.25, -0.5])

    assert value.shape == (3, 4)
    assert np.allclose(value.sum(axis=1), 1.0)
    assert model.training


def test_emission_adapter_returns_state_feature_means_and_preserves_mode():
    model = _model()
    model.train()
    value = emission_context_tensor(model, [0.25, -0.5])

    assert value.shape == (3, 2)
    assert np.isfinite(value).all()
    assert model.training
