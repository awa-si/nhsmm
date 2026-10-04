from __future__ import annotations

from scripts.validate_learning_prediction import SCENARIOS, _run_one, _summarize


def test_learning_prediction_smoke_proves_fit_and_oos_prediction() -> None:
    rows = [_run_one(seed, SCENARIOS["moderate"], max_iter=20) for seed in (901, 902)]
    summary = _summarize(SCENARIOS["moderate"], rows)

    assert all(float(row["parameter_l1_change"]) > 0.0 for row in rows)
    assert all(float(row["ll_gain"]) > 0.0 for row in rows)
    assert all(float(row["oos_accuracy"]) > 0.80 for row in rows)
    assert all(not bool(row["oos_health"]["state_collapsed"]) for row in rows)
    assert all(bool(row["same_model_fingerprint"]) for row in rows)
    assert all(bool(row["different_data_fingerprint"]) for row in rows)
    assert all(row["variable_prediction_lengths"] == [53, 101] for row in rows)
    assert summary["passed"]


def test_learning_prediction_null_control_does_not_invent_recovery() -> None:
    row = _run_one(903, SCENARIOS["null"], max_iter=20)

    assert float(row["parameter_l1_change"]) > 0.0
    assert float(row["oos_accuracy"]) < 0.50
    assert abs(float(row["oos_ari"])) < 0.10
    assert bool(row["same_model_fingerprint"])
    assert bool(row["different_data_fingerprint"])
