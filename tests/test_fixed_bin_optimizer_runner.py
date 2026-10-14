import copy
import json

import numpy as np
import pytest
import torch

import scripts.compare_fixed_bin_optimizers as driver
from scripts.fixed_bin_posterior_score import density
from scripts.verify_isolated_trading_hsmm import sequences


def method():
    return json.loads(driver.PROTOCOL.read_text())["method"]


def checkpoint():
    z = torch.zeros(4, 32, dtype=torch.float64)
    return {
        "pmf": density(z).tolist(),
        "logits": z.tolist(),
        "history": [
            {
                "completed_updates": 0,
                "ll": 0.0,
                "penalized_objective": 0.0,
                "gradient_max": 0.0,
                "elapsed_seconds": 0.0,
            }
        ],
        "adam_state": {"step": 0, "exp_avg": z.tolist(), "exp_avg_sq": z.tolist()},
    }


def test_exact_interrupted_adam_moments_and_updates():
    torch.set_num_threads(1)
    x, _ = sequences(401922, count=2, steps=60, sigma=2.0)
    p = np.full((4, 260), 1 / 260)
    complete = driver.fit_adam(x, p, 7, method())
    saved = []

    def interrupt(payload):
        if payload["history"][-1]["completed_updates"] == 5:
            saved.append(copy.deepcopy(payload))
            raise RuntimeError("planned interruption")

    with pytest.raises(RuntimeError, match="planned interruption"):
        driver.fit_adam(x, p, 7, method(), checkpoint=interrupt)
    resumed = driver.fit_adam(x, saved[0]["pmf"], 7, method(), record=saved[0])
    np.testing.assert_allclose(resumed["pmf"], complete["pmf"], atol=1e-15, rtol=0)
    assert resumed["adam_state"] == complete["adam_state"]
    assert [h["ll"] for h in resumed["history"]] == [h["ll"] for h in complete["history"]]
    terminal = driver.fit_adam(x, resumed["pmf"], 7, method(), record=resumed)
    assert terminal["adam_state"] == resumed["adam_state"]


@pytest.mark.parametrize("kind", ["step", "moment", "density"])
def test_corrupt_adam_checkpoint_rejected(kind):
    r = checkpoint()
    if kind == "step":
        r["adam_state"]["step"] = 1
    elif kind == "moment":
        r["adam_state"]["exp_avg_sq"][0][0] = -1
    else:
        r["logits"][0][0] = 0.1
    with pytest.raises(ValueError):
        driver.check_adam(r, 27)


def test_actual_bounded_phases_matched_start_and_evaluation(tmp_path):
    torch.set_num_threads(1)
    r = driver.expected_payload(1922, "unit-test")
    r["contract"]["data"].update(training_sequences=2, outer_sequences=2, steps=60)
    r["contract"]["method"].update(shared_warm_em_updates=1, updates_per_arm=2)
    x, _ = sequences(401922, count=2, steps=60, sigma=2.0)
    path = tmp_path / "checkpoint.json"
    assert driver.train(r, x, path, max_phases=1) is False
    assert list(r["phases"]) == ["warm"]
    assert r["evaluation"] is None
    loaded = driver.load_checkpoint(path, r)
    assert driver.train(loaded, x, path) is True
    assert loaded["phases"]["fixed_bin_em"]["history"][0]["ll"] == pytest.approx(
        loaded["phases"]["warm"]["history"][-1]["ll"], abs=1e-7
    )
    assert loaded["phases"]["fixed_bin_adam"]["history"][0]["ll"] == pytest.approx(
        loaded["phases"]["warm"]["history"][-1]["ll"], abs=1e-7
    )
    result = driver.evaluate(loaded, x)
    assert result["actual_updates"] == 5
    assert result["actual_adam_updates"] == 2
    assert result["labels_used_in_learning_or_selection"] is False
    assert list(result["arms"]) == ["fixed_bin_em", "fixed_bin_adam"]


@pytest.mark.parametrize("kind", ["source", "prefix", "early_outer"])
def test_resume_identity_and_phase_guards(tmp_path, kind):
    expected = driver.expected_payload(1922, "unit-test")
    r = copy.deepcopy(expected)
    if kind == "source":
        r["execution"]["source_commit"] = "wrong"
    elif kind == "prefix":
        r["phases"] = {"fixed_bin_adam": {}}
    else:
        r["evaluation"] = {}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        driver.load_checkpoint(path, expected)
