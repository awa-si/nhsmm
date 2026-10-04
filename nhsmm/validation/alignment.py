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
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class StateAlignment:
    """Permutation aligning candidate latent states to reference latent states."""

    candidate_to_reference: tuple[int, ...]
    cost: float

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
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("state centers must be finite")

    cost = ((cand[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
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
