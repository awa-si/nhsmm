from __future__ import annotations

from dataclasses import dataclass
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
    n = len(observations)
    if len(context) != n:
        raise ValueError("observations and context must have equal first-axis length")
    if n < 2:
        raise ValueError("at least two independent training units are required")

    split = n // 2 if split_index is None else int(split_index)
    if split <= 0 or split >= n:
        raise ValueError("split_index must leave both replicas non-empty")

    xa, ca = _slice(observations, 0, split), _slice(context, 0, split)
    xb, cb = _slice(observations, split, n), _slice(context, split, n)
    model_a = fit_model(xa, ca, 0)
    model_b = fit_model(xb, cb, 1)

    evidence = evaluate_context_replication(
        model_a,
        model_b,
        evidence_contexts,
        effect_matrix=effect_matrix,
        config=config,
        centers=centers,
    )
    return SplitFitEvidence(
        evidence=evidence,
        split_index=split,
        replica_sizes=(split, n - split),
    )
