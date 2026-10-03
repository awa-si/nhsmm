from __future__ import annotations

import numpy as np
import torch

from nhsmm.config import ModelConfig
from nhsmm.models.base import NHSMM
from nhsmm.validation import ContextEvidenceConfig
from nhsmm.validation.nhsmm import (
    evaluate_transition_context_replication,
    split_fit_transition_context_evidence,
    transition_context_matrix,
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
    model.eval()
    return model


def _policy() -> ContextEvidenceConfig:
    return ContextEvidenceConfig(replication_corr_min=-1.0, min_replica_amplitude=0.0)


def test_transition_context_matrix_is_a_duration_weighted_probability_matrix():
    model = _model()
    ctx = np.array([0.25, -0.5], dtype=np.float32)
    actual = transition_context_matrix(model, ctx)

    ct = torch.tensor(ctx).view(1, 1, 2)
    with torch.inference_mode():
        duration = model.dist.duration.log_matrix(
            context=ct, T=1, soft_dmax=model.duration_logits_bias
        ).exp()[0, 0]
        transition = model.dist.transition.log_matrix(context=ct, T=1).exp()[0, 0]
        expected = (duration.unsqueeze(-1) * transition).sum(dim=1).cpu().numpy()

    assert actual.shape == (3, 3)
    assert np.allclose(actual, expected)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_transition_context_matrix_preserves_training_mode():
    model = _model()
    model.train()
    transition_context_matrix(model, [0.0, 0.0])
    assert model.training


def test_transition_replication_api_is_state_label_semantic_not_internal_layer_semantic():
    a = _model(11)
    b = _model(11)
    b.load_state_dict(a.state_dict())

    result = evaluate_transition_context_replication(
        a,
        b,
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
        config=_policy(),
    )

    assert result.selected
    assert result.replication_corr >= 0.999
    assert result.n_contexts == 3


def test_split_fit_transition_api_keeps_training_policy_in_callback():
    observations = np.arange(48, dtype=np.float32).reshape(6, 4, 2)
    contexts = np.zeros((6, 4, 2), dtype=np.float32)
    seen: list[tuple[int, int]] = []
    template = _model(21)

    def fit_model(x, c, replica):
        seen.append((replica, len(x)))
        m = _model(21 + replica)
        m.load_state_dict(template.state_dict())
        return m

    result = split_fit_transition_context_evidence(
        observations,
        contexts,
        fit_model=fit_model,
        evidence_contexts=[[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        config=_policy(),
    )

    assert result.selected
    assert seen == [(0, 3), (1, 3)]
    assert result.replica_sizes == (3, 3)
