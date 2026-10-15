from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def finite_correlation(a: Any, b: Any) -> float:
    """Return Pearson correlation, using zero for degenerate finite vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("correlation inputs must have equal shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("correlation inputs must contain only finite values")
    if a.ndim != 1 or a.size == 0:
        raise ValueError("correlation inputs must be non-empty one-dimensional vectors")
    # Scale before centering: finite values can overflow variance/subtraction.
    a = a / max(float(np.max(np.abs(a))), np.finfo(np.float64).tiny)
    b = b / max(float(np.max(np.abs(b))), np.finfo(np.float64).tiny)
    a = a - a.mean()
    b = b - b.mean()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0))


@dataclass(frozen=True)
class StateAlignment:
    """Permutation aligning candidate latent states to reference latent states."""

    candidate_to_reference: tuple[int, ...]
    cost: float

    def __post_init__(self) -> None:
        permutation = self.candidate_to_reference
        if any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
            for x in permutation
        ) or sorted(permutation) != list(range(len(permutation))):
            raise ValueError("state alignment must be a permutation of 0..K-1")
        if not np.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("state alignment cost must be finite and non-negative")

    @property
    def reference_to_candidate(self) -> tuple[int, ...]:
        inverse = np.empty(len(self.candidate_to_reference), dtype=np.int64)
        for candidate, reference in enumerate(self.candidate_to_reference):
            inverse[reference] = candidate
        return tuple(int(x) for x in inverse)


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Return row->column minimum-cost assignment for a square cost matrix."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    n = cost.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(cost).all():
        raise ValueError("cost must contain only finite values")

    # Bound intermediate potentials even for finite costs near float64 limits.
    cost = cost / max(float(np.max(np.abs(cost))), 1.0)
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        row_to_col[p[j] - 1] = j - 1
    return row_to_col


def align_state_centers(reference: Any, candidate: Any) -> StateAlignment:
    """Align candidate state centers to reference centers by minimum squared distance."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape:
        raise ValueError("reference and candidate centers must have equal shape [K, F]")
    if 0 in ref.shape:
        raise ValueError("state centers must have non-empty shape [K, F]")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("state centers must be finite")

    try:
        with np.errstate(over="raise", invalid="raise"):
            cost = ((cand[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
    except FloatingPointError as exc:
        raise ValueError("state centers must yield finite squared distances") from exc
    permutation = _hungarian(cost)
    total_cost = float(cost[np.arange(cost.shape[0]), permutation].sum())
    return StateAlignment(tuple(int(x) for x in permutation), total_cost)


def emission_centers(model: Any) -> np.ndarray:
    """Extract semantic emission centers from a fitted NHSMM-like model."""
    try:
        emission = model.dist.emission
    except AttributeError as exc:
        raise TypeError(
            "model must expose dist.emission or provide a custom centers function"
        ) from exc

    value = None
    for name in ("mu", "loc", "base"):
        if hasattr(emission, name):
            value = getattr(emission, name)
            value = value() if callable(value) else value
            break
    if value is None:
        raise TypeError("emission must expose mu, loc, or base")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    centers = np.asarray(value, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError("emission centers must have shape [K, F]")
    return centers - centers.mean(axis=0, keepdims=True)


def reorder_square_matrix(matrix: Any, alignment: StateAlignment) -> np.ndarray:
    """Return a candidate matrix reordered into reference state coordinates."""
    matrix = np.asarray(matrix, dtype=np.float64)
    k = len(alignment.candidate_to_reference)
    if matrix.shape != (k, k):
        raise ValueError(f"matrix must have shape {(k, k)}")
    inverse = np.asarray(alignment.reference_to_candidate, dtype=np.int64)
    return matrix[np.ix_(inverse, inverse)]
