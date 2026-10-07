from __future__ import annotations

from scripts.validate_duration_tail import _acceptance


def test_duration_tail_acceptance_passes_observed_contract() -> None:
    true_tail = {
        "positive_brier_gain": 8,
        "median_brier_gain": 0.058,
        "positive_mae_gain": 8,
        "median_mae_gain": 0.016,
        "positive_ll_gain": 6,
        "median_ll_gain": 0.002,
        "median_learned_h_tail": 0.25,
    }
    finite_control = {
        "min_learned_h_tail": 1.0,
        "median_ll_gain": 0.00008,
        "median_mae_gain": -0.00018,
        "median_brier_gain": -0.00009,
    }

    result = _acceptance(true_tail, finite_control)

    assert result["passed"] is True
    assert all(result.values())


def test_duration_tail_acceptance_rejects_missing_tail_benefit() -> None:
    true_tail = {
        "positive_brier_gain": 4,
        "median_brier_gain": 0.005,
        "positive_mae_gain": 4,
        "median_mae_gain": 0.002,
        "positive_ll_gain": 4,
        "median_ll_gain": -0.001,
        "median_learned_h_tail": 0.8,
    }
    finite_control = {
        "min_learned_h_tail": 0.7,
        "median_ll_gain": 0.01,
        "median_mae_gain": 0.01,
        "median_brier_gain": 0.01,
    }

    result = _acceptance(true_tail, finite_control)

    assert result["passed"] is False
    assert not result["true_tail_calibration"]
    assert not result["true_tail_likelihood"]
    assert not result["true_tail_hazard_activates"]
    assert not result["finite_control_reduces_to_finite"]
