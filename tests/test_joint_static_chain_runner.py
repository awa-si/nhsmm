import copy
import json

import numpy as np
import pytest

from scripts.compare_adaptive_sample_size import save
from scripts.compare_joint_static_chain import (
    ORDER,
    check_record,
    expected_payload,
    fit_arm,
    initial_states,
    load_checkpoint,
    seal,
    train,
)
from scripts.static_chain_check import numerical_check
from scripts.verify_isolated_trading_hsmm import sequences


def problem():
    x, _ = sequences(401929, count=3, steps=60, sigma=2.0)
    return x, initial_states(x)


def test_exact_resume_and_terminal_noop():
    x, initials = problem()
    key = ORDER[2]
    full, _ = fit_arm(x, key, initials[key], 4)
    partial, used = fit_arm(x, key, initials[key], 4, max_new=2)
    assert used == 2
    resumed, used = fit_arm(x, key, initials[key], 4, record=partial)
    assert used == 2
    for k in ("pmf", "means", "variances", "transition", "initial_probabilities"):
        np.testing.assert_array_equal(full[k], resumed[k])
    assert [h["ll"] for h in full["history"]] == [h["ll"] for h in resumed["history"]]
    noop, used = fit_arm(x, key, initials[key], 4, record=resumed)
    assert used == 0 and noop == resumed


def test_frozen_emissions_and_actual_public_replay():
    x, initials = problem()
    for key in ORDER:
        result, used = fit_arm(x, key, initials[key], 2)
        assert used == 2
        if key != ORDER[2]:
            assert result["transition"] == initials[key]["transition"]
            assert result["initial_probabilities"] == initials[key]["initial_probabilities"]
        checks = numerical_check(
            x,
            *(
                np.asarray(result[k])
                for k in ("pmf", "means", "variances", "transition", "initial_probabilities")
            ),
        )
        assert max(checks.values()) <= 1e-7
        assert max(v for k, v in checks.items() if "filter" in k) <= 1e-8


@pytest.mark.parametrize(
    "kind", ["pmf", "mean", "history", "initial", "transition", "probabilities", "digest"]
)
def test_corrupt_checkpoint_rejected(kind):
    x, initials = problem()
    result, _ = fit_arm(x, ORDER[2], initials[ORDER[2]], 2)
    bad = copy.deepcopy(result)
    if kind == "pmf":
        bad["pmf"][0][0] *= 2
    elif kind == "mean":
        bad["means"][0] += 1
    elif kind == "history":
        bad["history"][-1]["completed_updates"] += 1
    elif kind == "transition":
        bad["transition"][0][1] += 0.1
    elif kind == "probabilities":
        bad["initial_probabilities"][0] += 0.1
    elif kind == "initial":
        bad["initial"]["means"][0] += 1
    else:
        bad["state_sha256"] = "wrong"
    with pytest.raises(ValueError):
        check_record(bad, 2)


def test_resealed_terminal_likelihood_tamper_rejected():
    x, initials = problem()
    record, _ = fit_arm(x, ORDER[2], initials[ORDER[2]], 2)
    record["history"][-1]["ll"] += 1
    seal(record)
    with pytest.raises(ValueError, match="terminal likelihood"):
        fit_arm(x, ORDER[2], initials[ORDER[2]], 2, record=record)


def test_header_prefix_and_early_evaluation_rejected(tmp_path):
    r = expected_payload(1934, "test-source")
    expected = copy.deepcopy(r)
    path = tmp_path / "checkpoint.json"
    r["execution"]["source_commit"] = "drift"
    save(path, r)
    with pytest.raises(ValueError, match="drift"):
        load_checkpoint(path, expected)
    r = copy.deepcopy(expected)
    r["evaluation"] = {}
    save(path, r)
    with pytest.raises(ValueError, match="premature"):
        load_checkpoint(path, expected)


def test_training_budget_checkpoint_and_completed_prefix(tmp_path):
    x, _ = problem()
    r = expected_payload(1934, "test-source")
    r["contract"]["fit"]["updates_per_arm"] = 2
    expected = copy.deepcopy(r)
    path = tmp_path / "checkpoint.json"
    assert not train(r, x, path, max_new=3)
    assert list(r["phases"]) == [ORDER[0]]
    assert len(r["active"]["history"]) - 1 == 1
    resumed = load_checkpoint(path, expected)
    assert train(resumed, x, path, max_new=3)
    assert sum(len(v["history"]) - 1 for v in resumed["phases"].values()) == 6
    assert json.loads(path.read_text())["active"] is None


def test_early_outer_guard_and_separate_chain_gates():
    from scripts.static_chain_evaluation import evaluate

    x, _ = problem()
    r = expected_payload(1934, "test-source")
    with pytest.raises(ValueError, match="requires all three"):
        evaluate(r, x)


def test_resealed_frozen_chain_drift_rejected():
    x, initials = problem()
    record, _ = fit_arm(x, ORDER[1], initials[ORDER[1]], 2)
    record["transition"][0][1] += 0.01
    record["transition"][0][2] -= 0.01
    record["history"][-1]["transition"] = record["transition"]
    seal(record)
    with pytest.raises(ValueError, match="frozen chain"):
        check_record(record, 2)
