"""Domain-neutral validation API for fitted NHSMM context effects."""

from .api import (
    ContextEffectComponent,
    context_effect,
    evaluate_context_effect_replication,
    split_fit_context_effect_evidence,
)
from .context_evidence import ContextEvidence, ContextEvidenceConfig
from .config import (
    VALIDATION_CONFIG_SCHEMA_VERSION,
    ValidationConfig,
    ValidationDataConfig,
    ValidationScenarioConfig,
)
from .split_fit import SplitFitEvidence
from .snapshot import (
    ValidationComparison,
    ValidationSnapshot,
    compare_validation_snapshots,
    evaluate_validation_snapshot,
    model_fingerprint,
)

__all__ = [
    "ContextEffectComponent",
    "ContextEvidence",
    "ContextEvidenceConfig",
    "VALIDATION_CONFIG_SCHEMA_VERSION",
    "ValidationConfig",
    "ValidationDataConfig",
    "ValidationScenarioConfig",
    "SplitFitEvidence",
    "context_effect",
    "evaluate_context_effect_replication",
    "split_fit_context_effect_evidence",
    "ValidationComparison",
    "ValidationSnapshot",
    "compare_validation_snapshots",
    "evaluate_validation_snapshot",
    "model_fingerprint",
]
