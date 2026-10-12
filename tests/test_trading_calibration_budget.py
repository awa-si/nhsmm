"""Excluded-window mean residuals, nested head selection, and optimizer checkpoint."""

import numpy as np
import pytest
import torch

from nhsmm import ModelConfig
from scripts.diagnose_trading_calibration_budget import (
    RIDGES,
    BudgetCheckpointModel,
    fit_crossfit_head,
    oof_mean_raw,
    target_returns,
    tune_head,
    window_folds,
)
from scripts.diagnose_trading_information_baselines import score_head


def test_window_folds_are_disjoint_complete_and_oof_predictions_exclude_own_labels():
    rng = np.random.default_rng(1)
    design = np.concatenate([np.ones((8, 6, 1)), rng.normal(size=(8, 6, 2))], axis=-1)
    target = rng.normal(0, 0.002, (8, 6))
    folds = window_folds(8)
    np.testing.assert_array_equal(np.concatenate(folds), np.arange(8))
    predictions = oof_mean_raw(design, target, 0.01)
    changed = target.copy()
    changed[:2] += 100
    np.testing.assert_array_equal(predictions[:2], oof_mean_raw(design, changed, 0.01)[:2])
    with pytest.raises(ValueError, match="two windows"):
        window_folds(1)


def test_nested_selection_uses_only_training_and_records_deterministic_candidates():
    rng = np.random.default_rng(2)
    design = np.concatenate([np.ones((8, 6, 1)), rng.normal(size=(8, 6, 2))], axis=-1)
    target = rng.normal(0, 0.002, (8, 6))
    head, evidence = tune_head(design, target)
    assert [c["ridge"] for c in evidence["candidate_scores"]] == list(RIDGES)
    assert all(len(c["fold_log_scores"]) == 4 for c in evidence["candidate_scores"])
    winner = max(evidence["candidate_scores"], key=lambda row: row["mean_log_score"])
    assert evidence["selected_ridge"] == winner["ridge"]
    independently_fitted = fit_crossfit_head(design, target, winner["ridge"])
    for key in head:
        np.testing.assert_array_equal(head[key], independently_fitted[key])
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in head.items()}
    heldout = rng.normal(0, 0.002, (2, 6))
    score_head(design[:2].reshape(-1, 3), target_returns(heldout), head)
    heldout[:] = 100
    score_head(design[:2].reshape(-1, 3), target_returns(heldout), head)
    for key in head:
        np.testing.assert_array_equal(head[key], before[key])


def test_crossfit_raw_unit_score_matches_independent_normal_density():
    rng = np.random.default_rng(3)
    design = np.concatenate([np.ones((8, 5, 1)), rng.normal(size=(8, 5, 2))], axis=-1)
    target = rng.normal(0, 0.002, (8, 5))
    head = fit_crossfit_head(design, target, 0.1)
    x = design.reshape(-1, 3)
    variance = np.maximum(head["smear"] * np.exp(np.clip(x @ head["gamma"], -12, 12)), 1e-6)
    distribution = torch.distributions.Normal(
        torch.tensor(head["mean"] + head["scale"] * (x @ head["beta"])),
        torch.tensor(head["scale"] * np.sqrt(variance)),
    )
    actual = score_head(x, target_returns(target), head)
    assert actual["pairs"] == 40
    assert actual["next_return_log_score"] == pytest.approx(
        float(distribution.log_prob(torch.tensor(target.reshape(-1))).mean()), abs=1e-12
    )


@pytest.mark.parametrize("family", ["gaussian", "studentt"])
def test_intermediate_checkpoint_matches_independently_stopped_production_fit(family):
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=4,
        causal=True,
        duration_tail=True,
        use_context_encoder=False,
        context_dim=1,
        n_init=1,
        convergence_stop=False,
        use_scheduler=False,
        emission_type=family,
        emission_init_mode="kmeans" if family == "studentt" else "kmeans_cluster_variance",
    )
    torch.manual_seed(21)
    x, c = torch.randn(2, 12, 2), torch.randn(2, 12, 1)
    snapshots = []
    for updates in (1, 3):
        np.random.seed(22)
        torch.manual_seed(22)
        model = BudgetCheckpointModel(
            cfg.with_overrides(max_iter=updates), device="cpu", checkpoint_iteration=1
        )
        model.initialize_distributions()
        model.optimize(x, context=c)
        snapshots.append(model.state_dict() if updates == 1 else model.checkpoint)
    assert snapshots[1] is not None
    assert snapshots[0].keys() == snapshots[1].keys()
    for key in snapshots[0]:
        torch.testing.assert_close(snapshots[0][key], snapshots[1][key], rtol=0, atol=0)
