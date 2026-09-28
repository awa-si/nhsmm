from .config import ModelConfig
from .encoder import DefaultEncoder
from .convergence import Convergence
from .models import NHSMM, DistributionSet
from .inference import load_inference_model, prepare_inference
from .artifact import build_artifact, load_artifact, save_artifact
from .benchmark import benchmark_cpu_runtime
from .filtering import HSMMFilterState
from .runtime import HSMMFilterRuntime, HSMMRuntimeState
from .validation import (
    ContextComponent,
    ContextEvidence,
    ContextEvidenceConfig,
    SplitFitEvidence,
    context_effect,
    duration_context_tensor,
    emission_context_tensor,
    evaluate_context_effect_replication,
    evaluate_duration_context_replication,
    evaluate_emission_context_replication,
    evaluate_initial_context_replication,
    evaluate_state_context_replication,
    evaluate_transition_context_replication,
    initial_context_tensor,
    split_fit_context_effect_evidence,
    split_fit_duration_context_evidence,
    split_fit_emission_context_evidence,
    split_fit_initial_context_evidence,
    split_fit_state_context_evidence,
    split_fit_transition_context_evidence,
    transition_context_matrix,
)

__all__ = [
    'ModelConfig',
    'DistributionSet',
    'DefaultEncoder',
    'Convergence',
    'NHSMM',
    'HSMMFilterState',
    'HSMMFilterRuntime',
    'HSMMRuntimeState',
    'load_inference_model',
    'prepare_inference',
    'build_artifact',
    'load_artifact',
    'save_artifact',
    'benchmark_cpu_runtime',
    # Canonical context-validation API.
    'ContextComponent',
    'ContextEvidence',
    'ContextEvidenceConfig',
    'SplitFitEvidence',
    'context_effect',
    'evaluate_context_effect_replication',
    'split_fit_context_effect_evidence',
    # Compatibility/advanced helpers retained without behavior changes.
    'duration_context_tensor',
    'emission_context_tensor',
    'evaluate_duration_context_replication',
    'evaluate_emission_context_replication',
    'evaluate_initial_context_replication',
    'evaluate_state_context_replication',
    'evaluate_transition_context_replication',
    'initial_context_tensor',
    'split_fit_duration_context_evidence',
    'split_fit_emission_context_evidence',
    'split_fit_initial_context_evidence',
    'split_fit_state_context_evidence',
    'split_fit_transition_context_evidence',
    'transition_context_matrix',
]
