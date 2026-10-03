from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nhsmm.validation.context_evidence import (
    ContextEvidenceConfig,
    evaluate_context_replication,
)
from nhsmm.validation.split_fit import split_fit_context_evidence


@dataclass
class FakeModel:
    centers: np.ndarray
    base: np.ndarray
    slope: np.ndarray


def centers(model):
    return model.centers


def effect_matrix(model, context):
    x = float(np.asarray(context).reshape(-1)[0])
    logits = model.base + x * model.slope
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def permute_model(model, p):
    p = np.asarray(p)
    return FakeModel(
        model.centers[p],
        model.base[np.ix_(p, p)],
        model.slope[np.ix_(p, p)],
    )


def config():
    return ContextEvidenceConfig(
        replication_corr_min=0.55,
        min_replica_amplitude=0.05,
    )


def test_state_label_permutation_does_not_change_replication_semantics():
    centers_a = np.array([[-2.0], [0.0], [2.0]])
    base = np.array([[0.0, 0.5, -0.3], [-0.2, 0.0, 0.4], [0.3, -0.5, 0.0]])
    slope = np.array([[0.0, 1.2, -1.2], [-0.8, 0.0, 0.8], [0.6, -0.6, 0.0]])
    model_a = FakeModel(centers_a, base, slope)
    model_b = permute_model(model_a, [2, 0, 1])

    result = evaluate_context_replication(
        model_a,
        model_b,
        [-1.0, -0.5, 0.5, 1.0],
        effect_matrix=effect_matrix,
        config=config(),
        centers=centers,
    )

    assert result.selected
    assert result.replication_corr > 0.999
    assert result.min_replica_amplitude > 0.05


def test_directionally_inconsistent_replica_is_rejected():
    centers_a = np.array([[-1.0], [0.0], [1.0]])
    base = np.zeros((3, 3))
    slope = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [0.7, -0.7, 0.0]])
    model_a = FakeModel(centers_a, base, slope)
    model_b = FakeModel(centers_a.copy(), base.copy(), -slope)

    result = evaluate_context_replication(
        model_a,
        model_b,
        [-1.0, -0.5, 0.5, 1.0],
        effect_matrix=effect_matrix,
        config=config(),
        centers=centers,
    )

    assert not result.selected
    assert result.replication_corr < 0.0


def test_null_effect_is_rejected_even_when_replica_directions_match():
    centers_a = np.array([[-1.0], [0.0], [1.0]])
    base = np.zeros((3, 3))
    slope = np.zeros((3, 3))
    model_a = FakeModel(centers_a, base, slope)
    model_b = FakeModel(centers_a.copy(), base.copy(), slope.copy())

    result = evaluate_context_replication(
        model_a,
        model_b,
        [-1.0, 1.0],
        effect_matrix=effect_matrix,
        config=config(),
        centers=centers,
    )

    assert not result.selected
    assert result.min_replica_amplitude == 0.0


def test_split_fit_uses_disjoint_training_units_and_reports_split_semantics():
    observations = np.arange(24).reshape(6, 4)
    context = np.arange(12).reshape(6, 2)
    seen = []
    centers_a = np.array([[-1.0], [0.0], [1.0]])
    base = np.zeros((3, 3))
    slope = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [0.7, -0.7, 0.0]])

    def fit_model(x, c, replica):
        seen.append((replica, x.copy(), c.copy()))
        return FakeModel(centers_a.copy(), base.copy(), slope.copy())

    result = split_fit_context_evidence(
        observations,
        context,
        fit_model=fit_model,
        effect_matrix=effect_matrix,
        evidence_contexts=[-1.0, -0.5, 0.5, 1.0],
        config=config(),
        centers=centers,
    )

    assert result.selected
    assert result.split_index == 3
    assert result.replica_sizes == (3, 3)
    assert np.array_equal(seen[0][1], observations[:3])
    assert np.array_equal(seen[1][1], observations[3:])
    assert set(seen[0][1].reshape(-1)).isdisjoint(set(seen[1][1].reshape(-1)))
