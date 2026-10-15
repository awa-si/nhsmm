"""Public optimization telemetry is explicit, observed and independent."""

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


def model(**options):
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=3,
        causal=True,
        seed=823,
        dropout=0.0,
        verbose=False,
        use_scheduler=False,
        n_init=2,
        max_iter=6,
        **options,
    )
    value = NHSMM(cfg, device="cpu")
    value.initialize_distributions()
    return value


def test_observed_updates_and_selected_restart_are_public_and_independent():
    value = model(convergence_stop=True, plateau_window=2, plateau_tol=1e6)
    with pytest.raises(RuntimeError, match="No completed"):
        value.optimization_summary()
    value.optimize(torch.tensor([[0.2], [-0.1], [0.8], [0.3]]))
    summary = value.optimization_summary()
    assert summary["main_updates_per_run"] == [2, 2]
    assert summary["main_updates"] == 4
    assert summary["transition_refinement_updates"] == 0
    scores = summary["post_main_restart_scores"]
    assert summary["selected_run"] == max(range(2), key=scores.__getitem__)
    summary["main_updates_per_run"][0] = 999
    assert value.optimization_summary()["main_updates_per_run"] == [2, 2]


def test_failed_new_fit_invalidates_previous_completed_summary(monkeypatch):
    value = model(convergence_stop=False)
    x = torch.tensor([[0.2], [-0.1], [0.8], [0.3]])
    value.optimize(x)
    assert value.optimization_summary()["main_updates"] == 12

    def fail(*args, **kwargs):
        raise RuntimeError("failed update")

    monkeypatch.setattr(value, "_backward_training_loss", fail)
    with pytest.raises(RuntimeError, match="failed update"):
        value.optimize(x)
    with pytest.raises(RuntimeError, match="No completed"):
        value.optimization_summary()
