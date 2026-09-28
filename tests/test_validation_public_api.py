from __future__ import annotations

import numpy as np
import pytest

from nhsmm.config import ModelConfig
from nhsmm.models.base import NHSMM
from nhsmm.validation import ContextEvidenceConfig
from nhsmm.validation.api import (
    context_effect,
    evaluate_context_effect_replication,
    split_fit_context_effect_evidence,
)
from nhsmm.validation.nhsmm import (
    evaluate_transition_context_replication,
    transition_context_matrix,
)
from nhsmm.validation.state_effect import (
    duration_context_tensor,
    emission_context_tensor,
    evaluate_duration_context_replication,
    evaluate_emission_context_replication,
    evaluate_initial_context_replication,
    initial_context_tensor,
)


def _model(seed: int = 1) -> NHSMM:
    cfg = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=4,
        context_dim=2,
        hidden_dim=2,
        seed=seed,
        causal=True,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    return model


def _clone_model(model: NHSMM) -> NHSMM:
    clone = _model(seed=model.config.seed)
    clone.load_state_dict(model.state_dict())
    clone.train(model.training)
    return clone


def _policy() -> ContextEvidenceConfig:
    return ContextEvidenceConfig(replication_corr_min=-1.0, min_replica_amplitude=0.0)


def test_context_effect_dispatch_matches_component_helpers() -> None:
    model = _model()
    context = [0.25, -0.5]
    expected = {
        "initial": initial_context_tensor(model, context),
        "duration": duration_context_tensor(model, context),
        "emission": emission_context_tensor(model, context),
        "transition": transition_context_matrix(model, context),
    }
    for component, value in expected.items():
        actual = context_effect(model, context, component=component)
        assert np.allclose(actual, value)


def test_replication_dispatch_matches_component_helpers() -> None:
    a = _model(seed=3)
    b = _clone_model(a)
    contexts = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    specialized = {
        "initial": evaluate_initial_context_replication(a, b, contexts, config=_policy()),
        "duration": evaluate_duration_context_replication(a, b, contexts, config=_policy()),
        "emission": evaluate_emission_context_replication(a, b, contexts, config=_policy()),
        "transition": evaluate_transition_context_replication(a, b, contexts, config=_policy()),
    }
    for component, expected in specialized.items():
        actual = evaluate_context_effect_replication(
            a, b, contexts, component=component, config=_policy()
        )
        assert actual == expected


def test_split_fit_dispatch_preserves_replica_contract() -> None:
    template = _model(seed=5)
    seen = []

    def fit_model(x, c, replica_index):
        seen.append((len(x), len(c), replica_index))
        return _clone_model(template)

    result = split_fit_context_effect_evidence(
        np.zeros((4, 2, 2)),
        np.zeros((4, 2, 2)),
        component="initial",
        fit_model=fit_model,
        evidence_contexts=[[-1.0, -1.0], [1.0, 1.0]],
        config=_policy(),
    )

    assert result.replica_sizes == (2, 2)
    assert seen == [(2, 2, 0), (2, 2, 1)]


def test_normalized_api_rejects_unknown_component() -> None:
    model = _model()
    with pytest.raises(ValueError, match="unknown context component"):
        context_effect(model, [0.0, 0.0], component="unknown")  # type: ignore[arg-type]
