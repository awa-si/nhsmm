from __future__ import annotations

import json

import pytest
import torch

from nhsmm import (
    ModelConfig,
    NHSMM,
    compare_validation_snapshots,
    evaluate_validation_snapshot,
    model_fingerprint,
)


def _model(seed: int = 41) -> NHSMM:
    model = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=5,
            causal=True,
            dropout=0.0,
            seed=seed,
            max_iter=1,
            use_scheduler=False,
            convergence_stop=False,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    return model


def test_validation_snapshot_is_observational_deterministic_and_json_serializable() -> None:
    torch.manual_seed(7)
    model = _model()
    model.train()
    x = [torch.randn(4, 2), torch.randn(7, 2)]

    first = evaluate_validation_snapshot(model, x, label="reference")
    second = evaluate_validation_snapshot(model, x, label="reference")

    assert model.training
    assert first.as_dict() == second.as_dict()
    assert first.sequence_lengths == (4, 7)
    assert first.n_timesteps == 11
    assert first.n_sequences == 2
    assert first.model_fingerprint == model_fingerprint(model)
    assert first.context_fingerprint is None
    assert 0.0 <= first.state_switch_rate <= 1.0
    assert first.mean_run_length >= 1.0
    json.dumps(first.as_dict(), sort_keys=True)


def test_validation_snapshot_fingerprints_data_and_explicit_context() -> None:
    torch.manual_seed(9)
    model = _model()
    x = torch.randn(2, 6, 2)
    context = torch.randn(2, 6, int(model.context_dim))

    first = evaluate_validation_snapshot(model, x, context=context)
    changed_data = x.clone()
    changed_data[0, 0, 0] += 1.0
    second = evaluate_validation_snapshot(model, changed_data, context=context)
    changed_context = context.clone()
    changed_context[0, 0, 0] += 1.0
    third = evaluate_validation_snapshot(model, x, context=changed_context)

    assert first.data_fingerprint != second.data_fingerprint
    assert first.context_fingerprint == second.context_fingerprint
    assert first.data_fingerprint == third.data_fingerprint
    assert first.context_fingerprint != third.context_fingerprint


def test_validation_comparison_reports_same_model_deltas() -> None:
    torch.manual_seed(13)
    model = _model()
    reference = evaluate_validation_snapshot(model, torch.randn(2, 6, 2), label="train")
    candidate = evaluate_validation_snapshot(model, torch.randn(2, 5, 2), label="oos")

    result = compare_validation_snapshots(reference, candidate)

    assert result.same_model
    assert result.reference_label == "train"
    assert result.candidate_label == "oos"
    assert result.occupancy_l1_distance >= 0.0
    json.dumps(result.as_dict(), sort_keys=True)


def test_validation_comparison_rejects_different_fitted_models_by_default() -> None:
    x = torch.randn(1, 6, 2)
    first = evaluate_validation_snapshot(_model(seed=1), x)
    second = evaluate_validation_snapshot(_model(seed=2), x)

    assert first.model_fingerprint != second.model_fingerprint
    with pytest.raises(ValueError, match="different model fingerprints"):
        compare_validation_snapshots(first, second)

    result = compare_validation_snapshots(first, second, require_same_model=False)
    assert not result.same_model


def test_model_fingerprint_changes_with_parameter_state() -> None:
    model = _model()
    before = model_fingerprint(model)
    with torch.no_grad():
        next(model.parameters()).add_(0.25)
    after = model_fingerprint(model)

    assert before != after


def test_validation_snapshot_rejects_empty_or_nonfinite_inputs() -> None:
    model = _model()
    with pytest.raises(ValueError, match="at least one valid timestep"):
        evaluate_validation_snapshot(model, torch.empty(1, 0, 2))

    x = torch.randn(1, 4, 2)
    x[0, 0, 0] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_validation_snapshot(model, x)


def test_model_fingerprint_supports_bfloat16_state() -> None:
    model = _model().to(dtype=torch.bfloat16)
    fingerprint = model_fingerprint(model)

    assert len(fingerprint) == 64
    assert fingerprint == model_fingerprint(model)
