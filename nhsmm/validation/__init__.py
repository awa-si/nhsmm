"""Domain-neutral validation utilities for fitted NHSMM context effects."""

from .alignment import StateAlignment, align_state_centers, emission_centers, reorder_square_matrix
from .api import (
    ContextComponent,
    context_effect,
    evaluate_context_effect_replication,
    split_fit_context_effect_evidence,
)
from .context_evidence import ContextEvidence, ContextEvidenceConfig, evaluate_context_replication
from .nhsmm import (
    evaluate_transition_context_replication,
    split_fit_transition_context_evidence,
    transition_context_matrix,
)
from .split_fit import SplitFitEvidence, split_fit_context_evidence
from .state_effect import (
    duration_context_tensor,
    emission_context_tensor,
    evaluate_duration_context_replication,
    evaluate_emission_context_replication,
    evaluate_initial_context_replication,
    evaluate_state_context_replication,
    initial_context_tensor,
    split_fit_duration_context_evidence,
    split_fit_emission_context_evidence,
    split_fit_initial_context_evidence,
    split_fit_state_context_evidence,
)

__all__ = [
    # Canonical public facade.
    "ContextComponent",
    "ContextEvidence",
    "ContextEvidenceConfig",
    "SplitFitEvidence",
    "context_effect",
    "evaluate_context_effect_replication",
    "split_fit_context_effect_evidence",
    # Advanced alignment/generic helpers.
    "StateAlignment",
    "align_state_centers",
    "emission_centers",
    "evaluate_context_replication",
    "evaluate_state_context_replication",
    "reorder_square_matrix",
    "split_fit_context_evidence",
    "split_fit_state_context_evidence",
    # Component-specific compatibility helpers.
    "duration_context_tensor",
    "emission_context_tensor",
    "evaluate_duration_context_replication",
    "evaluate_emission_context_replication",
    "evaluate_initial_context_replication",
    "evaluate_transition_context_replication",
    "initial_context_tensor",
    "split_fit_duration_context_evidence",
    "split_fit_emission_context_evidence",
    "split_fit_initial_context_evidence",
    "split_fit_transition_context_evidence",
    "transition_context_matrix",
]
