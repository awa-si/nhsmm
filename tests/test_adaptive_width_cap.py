"""Exact constrained partition optimum, observation-only fitting and recovery."""

import copy
import itertools
import json

import numpy as np
import pytest

from scripts.adaptive_width_cap import capped_mstep, fit_capped, refine_state
from scripts.compare_adaptive_duration_em import PSEUDOCOUNT, adaptive_mstep, fit_adaptive
from scripts.verify_isolated_trading_hsmm import sequences


@pytest.mark.parametrize("width", [1, 2, 3, 6])
@pytest.mark.parametrize("penalty", [0.0, 0.7, 8.0])
def test_capped_dp_matches_all_feasible_partitions(width, penalty):
    counts = np.array([[0.0, 0.0, 12.0, 15.0, 1.0, 1.0]])
    p, blocks = capped_mstep(counts, penalty, width)
    q = counts + PSEUDOCOUNT
    actual = float((q * np.log(p)).sum()) - penalty * (len(blocks[0]) - 1)
    best = -np.inf
    for flags in itertools.product((False, True), repeat=5):
        cuts = [0] + [i + 1 for i, split in enumerate(flags) if split] + [6]
        if any(b - a > width for a, b in zip(cuts, cuts[1:])):
            continue
        candidate = np.empty_like(p)
        for a, b in zip(cuts, cuts[1:]):
            candidate[0, a:b] = q[0, a:b].sum() / q.sum() / (b - a)
        best = max(best, float((q * np.log(candidate)).sum()) - penalty * (len(cuts) - 2))
    assert actual == pytest.approx(best, abs=1e-10)
    assert all(b - a <= width for a, b in blocks[0]) and np.all(p > 0)
    np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=1e-12)


def test_no_effect_when_constraint_inactive_and_zero_exposure_valid():
    counts = np.array([[0.0, 0.0, 9.0, 12.0, 1.0, 1.0]])
    a, blocks = capped_mstep(counts, 0.7, 6)
    b, other = adaptive_mstep(counts, 0.7)
    np.testing.assert_array_equal(a, b)
    assert blocks == other
    p, parts = capped_mstep(np.zeros((1, 19)), 5.0, 8)
    np.testing.assert_allclose(p, 1 / 19, rtol=0, atol=1e-15)
    assert all(end - start <= 8 for start, end in parts[0])


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_invalid_width_rejected(width):
    with pytest.raises(ValueError):
        capped_mstep([[1.0]], 1.0, width)


def test_warm_refinement_preserves_every_probability_and_input():
    p = np.full((4, 260), 1 / 260)
    blocks = [[(0, 260)]] * 4
    original = p.copy()
    refined, parts = refine_state(p, blocks, 8)
    np.testing.assert_array_equal(refined, original)
    np.testing.assert_array_equal(p, original)
    assert blocks == [[(0, 260)]] * 4
    assert parts[0][-1] == (256, 260) and len(parts[0]) == 33


def test_actual_censored_em_resume_matches_and_rejects_objective_tampering():
    x, _ = sequences(511, count=2, steps=60, sigma=2.0)
    p, b, _ = fit_adaptive(x, 0.0, iterations=2)
    initial = refine_state(p, b, 8)
    full, _, history = fit_capped(x, 0.2, 8, iterations=7, initial_state=initial)
    assert min(np.diff([z["penalized_objective"] for z in history])) >= -1e-6
    saved = {}

    def interrupted(p, blocks, history):
        saved.update(p=p, blocks=blocks, history=history)
        if history[-1]["completed_updates"] == 5:
            raise RuntimeError("interruption")

    with pytest.raises(RuntimeError):
        fit_capped(x, 0.2, 8, iterations=7, initial_state=initial, checkpoint=interrupted)
    resumed, _, h = fit_capped(
        x,
        0.2,
        8,
        iterations=7,
        initial_state=(saved["p"], saved["blocks"]),
        resume_history=saved["history"],
    )
    np.testing.assert_array_equal(full, resumed)
    np.testing.assert_allclose([z["ll"] for z in history], [z["ll"] for z in h], rtol=0, atol=1e-12)
    bad = copy.deepcopy(saved["history"])
    bad[-1]["ll"] += 1
    with pytest.raises(ValueError, match="reproduce"):
        fit_capped(
            x, 0.2, 8, iterations=7, initial_state=(saved["p"], saved["blocks"]), resume_history=bad
        )


def test_actual_driver_phase_budget_no_truth_in_fit_and_postfit_numerics(monkeypatch, tmp_path):
    import scripts.compare_adaptive_width_cap as pilot
    from scripts.compare_adaptive_sample_size import save

    contract = json.loads(pilot.PROTOCOL.read_text())
    contract["data"].update(training_sequences=2, outer_sequences=2, steps=60)
    contract["method"].update(warm_updates=1, penalized_updates=2)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(contract))
    monkeypatch.setattr(pilot, "PROTOCOL", protocol)
    expected = pilot.expected_payload(1909, "smoke")
    x, _ = sequences(401909, count=2, steps=60, sigma=2.0)
    path = tmp_path / "out.json"
    r = copy.deepcopy(expected)
    truth = pilot.true_law
    monkeypatch.setattr(pilot, "true_law", lambda: pytest.fail("truth entered fit"))
    with pytest.raises(ValueError, match="completed"):
        pilot.evaluate(r, x)
    assert not pilot.train(r, x, path, max_phases=1)
    assert list(r["phases"]) == ["warm"] and r["active"] is None
    r = pilot.load_checkpoint(path, expected)
    assert pilot.train(r, x, path, max_phases=2)
    assert list(r["phases"]) == ["warm", "unrestricted", "width8"]
    assert (
        r["phases"]["unrestricted"]["history"][0]["ll"] == r["phases"]["width8"]["history"][0]["ll"]
    )
    monkeypatch.setattr(pilot, "true_law", truth)
    r["evaluation"] = pilot.evaluate(r, x)
    save(path, r)
    assert r["evaluation"]["actual_msteps"] == 5
    for z in r["evaluation"]["arms"].values():
        assert max(z["gradient_check"].values()) < 1e-7
        assert z["public_recursion"]["maximum_state_probability_error"] < 1e-8
    import sys

    original = path.read_bytes()
    with monkeypatch.context() as guard:
        guard.setattr(
            sys,
            "argv",
            [
                "pilot",
                "--source-commit",
                "smoke",
                "--seed",
                "1909",
                "--output",
                str(path),
                "--resume",
            ],
        )
        guard.setattr(pilot, "train", lambda *a, **k: pytest.fail("completed resume fitted"))
        guard.setattr(pilot, "evaluate", lambda *a, **k: pytest.fail("completed resume evaluated"))
        pilot.main()
    assert path.read_bytes() == original
    corrupted = copy.deepcopy(r)
    corrupted["phases"]["warm"]["history"][0]["penalized_objective"] = (
        corrupted["phases"]["warm"]["history"][-1]["penalized_objective"] + 1
    )
    with pytest.raises(ValueError, match="history"):
        pilot.train(corrupted, x, path)
    bad = copy.deepcopy(expected)
    bad["seed"] = 1910
    with pytest.raises(ValueError, match="seed"):
        pilot.load_checkpoint(path, bad)
    tampered = copy.deepcopy(r)
    tampered["phases"] = {"width8": r["phases"]["width8"]}
    save(path, tampered)
    with pytest.raises(ValueError, match="prefix"):
        pilot.load_checkpoint(path, expected)


def test_frozen_protocol_source_identity_and_seed_guard(tmp_path):
    import scripts.compare_adaptive_width_cap as pilot
    from scripts.compare_adaptive_sample_size import save

    expected = pilot.expected_payload(1909, "test")
    path = tmp_path / "out.json"
    save(path, expected)
    other = copy.deepcopy(expected)
    other["execution"]["source_commit"] = "other"
    with pytest.raises(ValueError, match="execution"):
        pilot.load_checkpoint(path, other)
    other = copy.deepcopy(expected)
    other["contract"]["method"]["maximum_block_width"] = 16
    with pytest.raises(ValueError, match="contract"):
        pilot.load_checkpoint(path, other)
    with pytest.raises(ValueError, match="seed outside"):
        pilot.expected_payload(1906, "test")
