from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Callable, Iterable, Optional

from .alignment import emission_centers
from .context_evidence import ContextEvidence, ContextEvidenceConfig, evaluate_context_replication


@dataclass(frozen=True)
class SplitFitEvidence:
    evidence: ContextEvidence
    split_index: int
    replica_sizes: tuple[int, int]

    @property
    def selected(self) -> bool:
        return self.evidence.selected


def _slice(value: Any, start: int, stop: int) -> Any:
    try:
        return value[start:stop]
    except TypeError as exc:
        raise TypeError("observations and context must support first-axis slicing") from exc


def split_fit_context_evidence(
    observations: Any,
    context: Any,
    *,
    fit_model: Callable[[Any, Any, int], Any],
    effect_matrix: Callable[[Any, Any], Any],
    evidence_contexts: Iterable[Any],
    config: ContextEvidenceConfig,
    centers: Callable[[Any], Any] = emission_centers,
    split_index: Optional[int] = None,
) -> SplitFitEvidence:
    """Fit two disjoint replicas and evaluate semantic context-effect replication.

    ``fit_model(x_split, context_split, replica_index)`` returns a fitted model. The
    function deliberately does not call ``NHSMM.optimize`` itself, keeping training
    policy outside validation and preserving the core model API.
    """
    model_a, model_b, grid, split, n = _fit_disjoint_replicas(
        observations,
        context,
        fit_model=fit_model,
        evidence_contexts=evidence_contexts,
        split_index=split_index,
    )

    evidence = evaluate_context_replication(
        model_a,
        model_b,
        grid,
        effect_matrix=effect_matrix,
        config=config,
        centers=centers,
    )
    return SplitFitEvidence(
        evidence=evidence,
        split_index=split,
        replica_sizes=(split, n - split),
    )


def _fit_disjoint_replicas(
    observations: Any,
    context: Any,
    *,
    fit_model: Callable[[Any, Any, int], Any],
    evidence_contexts: Iterable[Any],
    split_index: Optional[int],
) -> tuple[Any, Any, list[Any], int, int]:
    n = len(observations)
    if len(context) != n:
        raise ValueError("observations and context must have equal first-axis length")
    if n < 2:
        raise ValueError("at least two independent training units are required")
    if split_index is None:
        split = n // 2
    else:
        if isinstance(split_index, bool):
            raise ValueError("split_index must be an integer")
        try:
            split = operator.index(split_index)
        except TypeError as exc:
            raise ValueError("split_index must be an integer") from exc
    if split <= 0 or split >= n:
        raise ValueError("split_index must leave both replicas non-empty")
    grid = list(evidence_contexts)
    if len(grid) < 2:
        raise ValueError("contexts must contain at least two grid points")

    model_a = fit_model(_slice(observations, 0, split), _slice(context, 0, split), 0)
    model_b = fit_model(_slice(observations, split, n), _slice(context, split, n), 1)
    if model_a is model_b:
        raise ValueError("fit_model must return distinct fitted model instances")
    return model_a, model_b, grid, split, n
