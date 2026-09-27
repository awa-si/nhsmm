from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nhsmm.config import ModelConfig
from nhsmm.models.base import NHSMM
from nhsmm.validation import (
    ContextEvidenceConfig,
    evaluate_context_replication,
    transition_context_matrix,
)


@dataclass
class FakeModel:
    centers: np.ndarray


def _centers(model: FakeModel) -> np.ndarray:
    return model.centers


def _model() -> NHSMM:
    cfg = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=4,
        context_dim=2,
        hidden_dim=2,
        seed=1,
        causal=True,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    return model


def test_context_evidence_policy_requires_finite_thresholds() -> None:
    with pytest.raises(ValueError, match="finite"):
        ContextEvidenceConfig(replication_corr_min=np.nan, min_replica_amplitude=0.1)
    with pytest.raises(ValueError, match="finite"):
        ContextEvidenceConfig(replication_corr_min=0.5, min_replica_amplitude=np.inf)


def test_context_evidence_rejects_non_finite_effect_matrices() -> None:
    model = FakeModel(np.array([[-1.0], [0.0], [1.0]]))

    def bad_effect_matrix(_model, _context):
        matrix = np.eye(3, dtype=float)
        matrix[0, 1] = np.nan
        return matrix

    with pytest.raises(ValueError, match="finite"):
        evaluate_context_replication(
            model,
            model,
            [-1.0, 1.0],
            effect_matrix=bad_effect_matrix,
            config=ContextEvidenceConfig(
                replication_corr_min=0.0,
                min_replica_amplitude=0.0,
            ),
            centers=_centers,
        )


def test_transition_context_matrix_rejects_non_finite_context() -> None:
    with pytest.raises(ValueError, match="finite"):
        transition_context_matrix(_model(), [np.nan, 0.0])


def test_transition_context_matrix_rejects_wrong_context_dimension() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        transition_context_matrix(_model(), [0.0])


def test_transition_context_matrix_requires_initialized_distributions() -> None:
    cfg = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=4,
        context_dim=2,
        hidden_dim=2,
        seed=1,
        causal=True,
    )
    model = NHSMM(cfg, device="cpu")
    with pytest.raises(TypeError, match="dist.duration and dist.transition"):
        transition_context_matrix(model, [0.0, 0.0])
