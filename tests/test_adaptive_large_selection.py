"""Research-driver recovery and training/selection boundaries."""

import copy
import json

import numpy as np
import pytest
import torch

import scripts.compare_adaptive_large_selection as driver
from scripts.compare_adaptive_sample_size import save
from scripts.verify_isolated_trading_hsmm import sequences


@pytest.fixture
def experiment():
    torch.set_num_threads(1)
    protocol = json.loads(driver.PROTOCOL.read_text())
    protocol["data"].update(
        training_sequences=8,
        inner_fit_sequences=6,
        inner_validation_sequences=2,
        outer_sequences=2,
        steps=35,
    )
    protocol["method"].update(warm_updates=1, penalized_updates=2, factors=[0.125, 2.0])
    expected = driver.expected_payload(protocol, 1906, "test-only")
    x, _ = sequences(401906, count=8, steps=35, sigma=2.0)
    return expected, x


@pytest.mark.parametrize("stop_update", [0, 2])
def test_real_interruption_resume_matches_uninterrupted_and_never_accesses_truth(
    experiment, tmp_path, monkeypatch, stop_update
):
    expected, x = experiment
    monkeypatch.setattr(driver, "true_law", lambda: pytest.fail("truth used in training"))
    # A validation tie must choose2 and share its completed refit with the control.
    monkeypatch.setattr(driver, "aggregate", lambda scores, p: (0.0, None))
    reference = copy.deepcopy(expected)
    assert driver.train(reference, x, tmp_path / "reference.json")
    fit = driver.fit_adaptive

    def interrupted(*args, **kwargs):
        if kwargs["tag"] == "candidate_0":
            callback = kwargs["checkpoint"]

            def stop(p, blocks, history):
                callback(p, blocks, history)
                if history[-1]["completed_updates"] == stop_update:
                    raise InterruptedError("simulated runtime loss after atomic checkpoint")

            kwargs["checkpoint"] = stop
        return fit(*args, **kwargs)

    path = tmp_path / "resume.json"
    partial = copy.deepcopy(expected)
    monkeypatch.setattr(driver, "fit_adaptive", interrupted)
    with pytest.raises(InterruptedError):
        driver.train(partial, x, path)
    restored = driver.load_checkpoint(path, expected)
    assert restored["active"]["key"] == "candidate_0"
    assert restored["evaluation"] is None
    monkeypatch.setattr(driver, "fit_adaptive", fit)
    assert driver.train(restored, x, path)
    assert restored["selection"]["selected_factor"] == 2
    assert "factor2_control" not in restored["phases"]
    assert not set(restored["split"]["train_indices"]) & set(
        restored["split"]["validation_indices"]
    )
    assert sum(p["updates"] for p in restored["phases"].values()) == 8
    for key, record in restored["phases"].items():
        np.testing.assert_array_equal(record["pmf"], reference["phases"][key]["pmf"])
        assert json.loads(json.dumps(record["blocks"])) == json.loads(
            json.dumps(reference["phases"][key]["blocks"])
        )
        for a, b in zip(record["history"], reference["phases"][key]["history"], strict=True):
            assert a["ll"] == pytest.approx(b["ll"], abs=1e-10)
            assert a["penalized_objective"] == pytest.approx(b["penalized_objective"], abs=1e-10)


def test_phase_limits_preserve_completed_fits_and_data_contract(experiment, tmp_path):
    expected, x = experiment
    payload = copy.deepcopy(expected)
    path = tmp_path / "bounded.json"
    assert not driver.train(payload, x, path, max_phases=1)
    assert list(payload["phases"]) == ["inner_warm"] and payload["active"] is None
    assert payload["selection"] is None and payload["evaluation"] is None
    warm_pmf = copy.deepcopy(payload["phases"]["inner_warm"]["pmf"])
    restored = driver.load_checkpoint(path, expected)
    assert not driver.train(restored, x, path, max_phases=1)
    assert list(restored["phases"]) == ["inner_warm", "candidate_0"]
    assert restored["phases"]["inner_warm"]["pmf"] == warm_pmf
    changed = x.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="replay differs"):
        driver.train(restored, changed, path)
    for key in ("seed", "contract", "execution"):
        wrong = copy.deepcopy(expected)
        wrong[key] = "different"
        with pytest.raises(ValueError, match=f"resume {key}"):
            driver.load_checkpoint(path, wrong)
    restored["phases"]["inner_warm"]["pmf"][0][0] += 0.1
    with pytest.raises(ValueError, match="normalized"):
        driver.train(restored, x, path)


def test_validation_only_selects_factor_and_distinct_control_is_fitted(
    experiment, tmp_path, monkeypatch
):
    expected, x = experiment
    seen = []

    def score(scores, p):
        seen.append(scores.copy())
        return -float(len(seen)), None  # Choose first factor, never truth.

    monkeypatch.setattr(driver, "aggregate", score)
    payload = copy.deepcopy(expected)
    assert driver.train(payload, x, tmp_path / "control.json")
    assert payload["selection"]["selected_factor"] == 0.125
    assert "factor2_control" in payload["phases"]
    assert sum(p["updates"] for p in payload["phases"].values()) == 10
    for scores in seen:
        np.testing.assert_array_equal(
            scores, driver.emission_scores(x[payload["split"]["validation_indices"]])
        )
    assert payload["evaluation"] is None


def test_completed_cli_resume_is_noop(experiment, tmp_path, monkeypatch):
    expected, _ = experiment
    expected["evaluation"] = {"completed": True}
    path = tmp_path / "done.json"
    save(path, expected)
    monkeypatch.setattr(driver, "expected_payload", lambda *args: copy.deepcopy(expected))
    monkeypatch.setattr(
        driver, "sequences", lambda *a, **k: pytest.fail("completed data regenerated")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "driver",
            "--seed",
            "1906",
            "--base-commit",
            "test-only",
            "--output",
            str(path),
            "--resume",
        ],
    )
    driver.main()


def test_incomplete_evaluation_rejected_before_outer_data_or_truth(experiment, monkeypatch):
    payload, x = experiment
    monkeypatch.setattr(driver, "sequences", lambda *a, **k: pytest.fail("outer data too early"))
    monkeypatch.setattr(driver, "true_law", lambda: pytest.fail("truth too early"))
    with pytest.raises(ValueError, match="requires completed"):
        driver.evaluate(payload, x)


def test_real_short_fit_and_public_evaluation_smoke(experiment, tmp_path):
    payload, x = experiment
    assert driver.train(payload, x, tmp_path / "smoke.json")
    result = driver.evaluate(payload, x)
    assert result["actual_msteps"] in (8, 10)
    assert not result["labels_used_in_learning_or_selection"]
    assert np.isfinite(result["uniform_heldout_ll"])
    for arm in result["results_by_phase"].values():
        assert np.isfinite(arm["heldout_ll"])
        assert arm["gradient_check"]["maximum_all_logit_gradient_error"] < 1e-7
        assert arm["public_recursion"]["ll_error"] < 1e-7
        assert arm["public_recursion"]["maximum_state_probability_error"] < 1e-7
