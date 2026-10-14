"""Actual resumed fixed-bin EM, guarded ledger and post-fit-only evaluator."""

import copy
import json

import numpy as np
import pytest
import torch

import scripts.compare_block_mass_smoothing as driver
from scripts.compare_adaptive_sample_size import save
from scripts.verify_isolated_trading_hsmm import sequences


def test_actual_interrupted_resume_exact_and_evidence_tampering_rejected():
    torch.set_num_threads(1)
    x, _ = sequences(822, count=2, steps=60, sigma=2.0)
    opts = driver.controls(json.loads(driver.PROTOCOL.read_text())["method"])
    full, h = driver.fit_fixed(x, np.log(1024), iterations=7, solver_controls=opts)
    saved = {}

    def interrupt(p, history):
        saved.update(p=p, history=history)
        if history[-1]["completed_updates"] == 5:
            raise RuntimeError("interruption")

    with pytest.raises(RuntimeError):
        driver.fit_fixed(x, np.log(1024), iterations=7, solver_controls=opts, checkpoint=interrupt)
    resumed, rh = driver.fit_fixed(
        x,
        np.log(1024),
        iterations=7,
        solver_controls=opts,
        initial_pmf=saved["p"],
        resume_history=saved["history"],
    )
    np.testing.assert_array_equal(full, resumed)
    np.testing.assert_allclose([z["ll"] for z in h], [z["ll"] for z in rh], rtol=0, atol=1e-12)
    bad = copy.deepcopy(saved["history"])
    bad[-1]["ll"] += 1
    with pytest.raises(ValueError, match="reproduce"):
        driver.fit_fixed(
            x,
            np.log(1024),
            iterations=7,
            solver_controls=opts,
            initial_pmf=saved["p"],
            resume_history=bad,
        )
    bad = copy.deepcopy(saved["history"])
    bad[-1]["solver"]["maximum_normalized_anchored_gradient"] = 1.0
    with pytest.raises(ValueError, match="certificate"):
        driver.fit_fixed(
            x,
            np.log(1024),
            iterations=7,
            solver_controls=opts,
            initial_pmf=saved["p"],
            resume_history=bad,
        )


def test_actual_cli_bounded_phases_shared_warm_postfit_evaluation_and_noop(monkeypatch, tmp_path):
    import sys

    contract = json.loads(driver.PROTOCOL.read_text())
    contract["data"].update(training_sequences=2, outer_sequences=2, steps=60)
    contract["method"].update(warm_updates=1, penalized_updates_per_arm=2)
    protocol = tmp_path / "contract.json"
    protocol.write_text(json.dumps(contract))
    monkeypatch.setattr(driver, "PROTOCOL", protocol)
    path = tmp_path / "out.json"
    argv = [
        "driver",
        "--source-commit",
        "smoke",
        "--seed",
        "1912",
        "--output",
        str(path),
        "--max-phases",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with monkeypatch.context() as guard:
        guard.setattr(
            driver, "evaluate_paired", lambda *a, **k: pytest.fail("early outer evaluation")
        )
        driver.main()
    r = json.loads(path.read_text())
    assert list(r["phases"]) == ["warm"] and r["evaluation"] is None and r["active"] is None
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["2", "--resume"])
    driver.main()
    r = json.loads(path.read_text())
    assert list(r["phases"]) == list(driver.ORDER) and r["active"] is None
    assert r["evaluation"]["actual_msteps"] == 5
    assert (
        r["phases"]["unsmoothed"]["history"][0]["ll"]
        == r["phases"]["smoothed"]["history"][0]["ll"]
        == r["phases"]["warm"]["history"][-1]["ll"]
    )
    assert r["phases"]["unsmoothed"]["blocks"] == r["phases"]["smoothed"]["blocks"]
    for z in r["evaluation"]["arms"].values():
        assert max(z["gradient_check"].values()) <= 1e-7
        assert z["public_recursion"]["maximum_state_probability_error"] <= 1e-8
    raw = path.read_bytes()
    with monkeypatch.context() as guard:
        guard.setattr(driver, "train", lambda *a, **k: pytest.fail("completed experiment fitted"))
        guard.setattr(
            driver, "evaluate", lambda *a, **k: pytest.fail("completed experiment evaluated")
        )
        driver.main()
    assert path.read_bytes() == raw


@pytest.mark.parametrize("key", ["execution", "seed", "contract"])
def test_frozen_identity_guard(key, tmp_path):
    expected = driver.expected_payload(1912, "source")
    path = tmp_path / "out.json"
    save(path, expected)
    other = copy.deepcopy(expected)
    other[key] = {"tampered": True}
    with pytest.raises(ValueError, match=key):
        driver.load_checkpoint(path, other)


def test_prefix_active_budget_data_and_premature_evaluation_guards(monkeypatch, tmp_path):
    expected = driver.expected_payload(1912, "source")
    path = tmp_path / "out.json"
    for bad in ({"phases": {"smoothed": {}}}, {"active": {"key": "smoothed"}}, {"evaluation": {}}):
        r = {**copy.deepcopy(expected), **bad}
        save(path, r)
        with pytest.raises(ValueError):
            driver.load_checkpoint(path, expected)
    x, _ = sequences(823, count=2, steps=60, sigma=2.0)
    r = copy.deepcopy(expected)
    r["contract"]["data"].update(training_sequences=2, steps=60)
    r["contract"]["method"]["warm_updates"] = 1
    assert not driver.train(r, x, path, max_phases=1)
    r["training_sha256"] = "bad"
    with pytest.raises(ValueError, match="training replay"):
        driver.train(r, x, path, max_phases=1)
    r["training_sha256"] = driver.digest(x)
    r["phases"]["warm"]["updates"] = 2
    with pytest.raises(ValueError, match="identity"):
        driver.train(r, x, path, max_phases=1)
    with pytest.raises(ValueError, match="completed"):
        driver.evaluate(expected, x)


def test_solver_protocol_controls_and_seed_fail_closed():
    m = json.loads(driver.PROTOCOL.read_text())["method"]
    m["solver_newton_polish_max_iterations"] = 4
    with pytest.raises(ValueError, match="solver"):
        driver.controls(m)
    with pytest.raises(ValueError, match="seed"):
        driver.expected_payload(1909, "source")
