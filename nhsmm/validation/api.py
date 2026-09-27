from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Optional

import numpy as np

from .context_evidence import ContextEvidence, ContextEvidenceConfig
from .nhsmm import (
    evaluate_transition_context_replication,
    split_fit_transition_context_evidence,
    transition_context_matrix,
)
from .split_fit import SplitFitEvidence
from .state_effect import (
    duration_context_tensor,
    emission_context_tensor,
    evaluate_duration_context_replication,
    evaluate_emission_context_replication,
    evaluate_initial_context_replication,
    initial_context_tensor,
    split_fit_duration_context_evidence,
    split_fit_emission_context_evidence,
    split_fit_initial_context_evidence,
)

ContextComponent = Literal["initial", "duration", "emission", "transition"]

_COMPONENTS = ("initial", "duration", "emission", "transition")


def _component_name(component: str) -> ContextComponent:
    if component not in _COMPONENTS:
        allowed = ", ".join(_COMPONENTS)
        raise ValueError(f"unknown context component {component!r}; expected one of: {allowed}")
    return component  # type: ignore[return-value]


def context_effect(
    model: Any,
    context: Any,
    *,
    component: ContextComponent,
) -> np.ndarray:
    """Return the semantic context-conditioned effect for one NHSMM component.

    Shapes are component-specific but stable:

    - ``initial`` -> ``[K]`` initial-state probabilities;
    - ``duration`` -> ``[K,D]`` duration probabilities;
    - ``emission`` -> ``[K,F]`` state-conditioned emission means;
    - ``transition`` -> ``[K,K]`` effective boundary-transition probabilities.
    """
    name = _component_name(component)
    if name == "initial":
        return initial_context_tensor(model, context)
    if name == "duration":
        return duration_context_tensor(model, context)
    if name == "emission":
        return emission_context_tensor(model, context)
    return transition_context_matrix(model, context)


def evaluate_context_effect_replication(
    model_a: Any,
    model_b: Any,
    contexts: Iterable[Any],
    *,
    component: ContextComponent,
    config: ContextEvidenceConfig,
) -> ContextEvidence:
    """Evaluate replicated context effects through one component-normalized API."""
    name = _component_name(component)
    if name == "initial":
        return evaluate_initial_context_replication(model_a, model_b, contexts, config=config)
    if name == "duration":
        return evaluate_duration_context_replication(model_a, model_b, contexts, config=config)
    if name == "emission":
        return evaluate_emission_context_replication(model_a, model_b, contexts, config=config)
    return evaluate_transition_context_replication(model_a, model_b, contexts, config=config)


def split_fit_context_effect_evidence(
    observations: Any,
    context: Any,
    *,
    component: ContextComponent,
    fit_model: Callable[[Any, Any, int], Any],
    evidence_contexts: Iterable[Any],
    config: ContextEvidenceConfig,
    split_index: Optional[int] = None,
) -> SplitFitEvidence:
    """Run split-fit context validation through one component-normalized API."""
    name = _component_name(component)
    kwargs = dict(
        fit_model=fit_model,
        evidence_contexts=evidence_contexts,
        config=config,
        split_index=split_index,
    )
    if name == "initial":
        return split_fit_initial_context_evidence(observations, context, **kwargs)
    if name == "duration":
        return split_fit_duration_context_evidence(observations, context, **kwargs)
    if name == "emission":
        return split_fit_emission_context_evidence(observations, context, **kwargs)
    return split_fit_transition_context_evidence(observations, context, **kwargs)
