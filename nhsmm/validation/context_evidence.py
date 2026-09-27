from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .alignment import StateAlignment, align_state_centers, emission_centers, reorder_square_matrix


@dataclass(frozen=True)
class ContextEvidenceConfig:
    """Selection policy for replicated context effects.

    Thresholds are explicit by design: validation policy is application/profile level,
    not part of NHSMM probabilistic semantics.
    """

    replication_corr_min: float
    min_replica_amplitude: float
    exclude_self_transitions: bool = True

    def __post_init__(self) -> None:
        if not -1.0 <= self.replication_corr_min <= 1.0:
            raise ValueError("replication_corr_min must be within [-1, 1]")
        if self.min_replica_amplitude < 0.0:
            raise ValueError("min_replica_amplitude must be non-negative")


@dataclass(frozen=True)
class ContextEvidence:
    selected: bool
    replication_corr: float
    direction_agreement: float
    replica_amplitudes: tuple[float, float]
    min_replica_amplitude: float
    state_alignment: StateAlignment
    n_contexts: int


def _conditional_change(matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(matrix, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("effect matrices must be square [K, K]")
    np.fill_diagonal(q, 0.0)
    rows = q.sum(axis=1, keepdims=True)
    if np.any(rows <= 0.0):
        raise ValueError("each state must retain positive off-diagonal mass")
    return q / rows


def _effect_vector(matrices: np.ndarray, *, exclude_self: bool) -> tuple[np.ndarray, float]:
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("context effect matrices must have shape [G, K, K]")
    if matrices.shape[0] < 2:
        raise ValueError("at least two context grid points are required")

    if exclude_self:
        matrices = np.stack([_conditional_change(m) for m in matrices])
        mask = ~np.eye(matrices.shape[1], dtype=bool)
    else:
        rows = matrices.sum(axis=-1)
        if np.any(rows <= 0.0):
            raise ValueError("matrix rows must have positive mass")
        matrices = matrices / rows[..., None]
        mask = np.ones((matrices.shape[1], matrices.shape[2]), dtype=bool)

    values = matrices[:, mask]
    centered = values - values.mean(axis=0, keepdims=True)
    amplitude = float(np.mean(np.ptp(values, axis=0)))
    return centered.reshape(-1), amplitude


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_context_replication(
    model_a: Any,
    model_b: Any,
    contexts: Iterable[Any],
    *,
    effect_matrix: Callable[[Any, Any], Any],
    config: ContextEvidenceConfig,
    centers: Callable[[Any], Any] = emission_centers,
) -> ContextEvidence:
    """Evaluate whether a learned context effect replicates across two fitted models.

    ``effect_matrix(model, context)`` must return a transition-like KxK matrix. The
    detector consumes fitted-model outputs only; latent truth labels are never required.
    """
    grid = list(contexts)
    if len(grid) < 2:
        raise ValueError("contexts must contain at least two grid points")

    alignment = align_state_centers(centers(model_a), centers(model_b))
    mats_a = []
    mats_b = []
    for context in grid:
        a = np.asarray(effect_matrix(model_a, context), dtype=np.float64)
        b = reorder_square_matrix(effect_matrix(model_b, context), alignment)
        if a.shape != b.shape:
            raise ValueError("replica effect matrices must have equal shape")
        mats_a.append(a)
        mats_b.append(b)

    vec_a, amp_a = _effect_vector(
        np.stack(mats_a), exclude_self=config.exclude_self_transitions
    )
    vec_b, amp_b = _effect_vector(
        np.stack(mats_b), exclude_self=config.exclude_self_transitions
    )
    replication_corr = _corr(vec_a, vec_b)
    nz = (np.abs(vec_a) > 1e-4) & (np.abs(vec_b) > 1e-4)
    agreement = float(((vec_a[nz] * vec_b[nz]) > 0.0).mean()) if np.any(nz) else 0.0
    amp_min = min(amp_a, amp_b)
    selected = (
        replication_corr >= config.replication_corr_min
        and amp_min >= config.min_replica_amplitude
    )
    return ContextEvidence(
        selected=bool(selected),
        replication_corr=replication_corr,
        direction_agreement=agreement,
        replica_amplitudes=(amp_a, amp_b),
        min_replica_amplitude=amp_min,
        state_alignment=alignment,
        n_contexts=len(grid),
    )
