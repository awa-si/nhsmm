"""Exact partition search, penalized EM semantics and observation-only selection."""

import itertools

import numpy as np
import pytest

from scripts.compare_adaptive_duration_em import (
    PSEUDOCOUNT,
    adaptive_mstep,
    fit_adaptive,
    objective,
    select_penalty,
)
from scripts.verify_isolated_trading_hsmm import sequences


@pytest.mark.parametrize("penalty", [0.0, 0.7, 8.0])
def test_dynamic_program_matches_exhaustive_partitions(penalty):
    counts = np.array([[0.0, 0.0, 12.0, 15.0, 1.0, 1.0]])
    p, blocks = adaptive_mstep(counts, penalty)
    q = counts + PSEUDOCOUNT
    actual = float((q * np.log(p)).sum()) - penalty * (len(blocks[0]) - 1)
    best = -np.inf
    for flags in itertools.product([False, True], repeat=counts.shape[1] - 1):
        cuts = [0] + [i + 1 for i, split in enumerate(flags) if split] + [counts.shape[1]]
        candidate = np.empty_like(p)
        for start, end in zip(cuts[:-1], cuts[1:]):
            candidate[0, start:end] = q[0, start:end].sum() / q.sum() / (end - start)
        score = float((q * np.log(candidate)).sum()) - penalty * (len(cuts) - 2)
        best = max(best, score)
    assert actual == pytest.approx(best, abs=1e-10)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
    assert np.all(p > 0)


def test_adaptive_boundaries_follow_counts_and_all_ages_remain_available():
    counts = np.array([[0.0] * 7 + [10.0] * 8 + [0.0] * 4])
    p, blocks = adaptive_mstep(counts, 1.0)
    assert blocks == [[(0, 7), (7, 15), (15, 19)]]
    assert np.all(p > 0) and p[0, 0] < p[0, 7]
    zero, zero_blocks = adaptive_mstep(np.zeros((1, 9)), 5.0)
    assert zero_blocks == [[(0, 9)]]
    np.testing.assert_allclose(zero, 1 / 9)


@pytest.mark.parametrize("counts,penalty", [([[np.nan]], 1), ([[-1]], 1), ([[1]], -1)])
def test_invalid_counts_or_penalty_rejected(counts, penalty):
    with pytest.raises(ValueError):
        adaptive_mstep(counts, penalty)


def test_penalized_observation_fit_without_truth_and_public_recursion(monkeypatch):
    import scripts.compare_adaptive_duration_em as adaptive
    from scripts.compare_compact_duration_em import public_check
    from scripts.compare_unlabelled_duration_em import posterior_gradient_check

    x, labels = sequences(301, count=2, steps=60, sigma=2.0)
    monkeypatch.setattr(adaptive, "true_law", lambda: pytest.fail("truth accessed while fitting"))
    p, blocks, history = fit_adaptive(x, 0.0, iterations=3)
    assert len(history) == 4
    assert min(np.diff([h["penalized_objective"] for h in history])) >= -1e-6
    assert history[-1]["penalized_objective"] > history[0]["penalized_objective"]
    assert np.isfinite(objective(history[-1]["ll"], p, blocks, 0.0))
    support = np.ones_like(p, dtype=bool)
    assert public_check(p, x, labels, support=support)["ll_error"] < 1e-7
    assert (
        posterior_gradient_check(x[:1], p, support=support)["maximum_all_logit_gradient_error"]
        < 1e-7
    )


def test_selection_receives_only_inner_training_observations(monkeypatch):
    import scripts.compare_adaptive_duration_em as adaptive

    x = np.arange(8 * 2).reshape(8, 2, 1)
    seen = []

    def fitted(train, penalty, **kwargs):
        seen.append(train.copy())
        return np.full((4, 260), 1 / 260), [[(0, 260)]] * 4, []

    monkeypatch.setattr(adaptive, "fit_adaptive", fitted)
    monkeypatch.setattr(adaptive, "aggregate", lambda scores, p: (0.0, None))
    factor, selected = select_penalty(x, seed=19)
    assert factor == 2.0  # Exact likelihood tie prefers stronger regularization.
    tr, va = selected["train_indices"], selected["validation_indices"]
    assert len(tr) == 6 and len(va) == 2 and not set(tr) & set(va)
    for train in seen:
        np.testing.assert_array_equal(train, x[tr])


def test_warm_start_preserves_input_and_checks_initial_partition():
    x, _ = sequences(302, count=2, steps=60, sigma=2.0)
    warm_p, warm_blocks, _ = fit_adaptive(x, 0.0, iterations=2)
    original = warm_p.copy()
    p, blocks, history = fit_adaptive(x, 0.2, iterations=2, initial_state=(warm_p, warm_blocks))
    np.testing.assert_array_equal(warm_p, original)
    assert min(np.diff([h["penalized_objective"] for h in history])) >= -1e-6
    assert np.all(p > 0) and len(blocks) == 4
    with pytest.raises(ValueError, match="cover"):
        fit_adaptive(x, 0.2, initial_state=(warm_p, [[(0, 259)]] * 4))
    with pytest.raises(ValueError, match="positive"):
        fit_adaptive(x, 0.2, initial_state=(np.zeros((4, 260)), warm_blocks))


def test_paired_selection_shares_inner_warm_and_matches_path_budgets(monkeypatch):
    import scripts.compare_adaptive_warmstart as paired

    x = np.arange(8 * 2).reshape(8, 2, 1)
    seen = []
    warm_p = np.full((4, 260), 1 / 260)
    warm_blocks = [[(0, 260)]] * 4

    def fitted(train, penalty, *, iterations, initial_state=None, **kwargs):
        seen.append((train.copy(), penalty, iterations, initial_state))
        return warm_p, warm_blocks, []

    monkeypatch.setattr(paired, "fit_adaptive", fitted)
    monkeypatch.setattr(paired, "aggregate", lambda scores, p: (0.0, None))
    result = paired.paired_selection(x, seed=20)
    assert result["selected_factor_by_mode"] == {"uniform": 2.0, "warm": 2.0}
    assert len(seen) == 7 and seen[0][2] == 3
    for train, penalty, updates, initial in seen:
        np.testing.assert_array_equal(train, x[result["train_indices"]])
    assert [s[2] for s in seen[1:4]] == [9] * 3
    for call in seen[4:]:
        assert call[2] == 6 and call[3][0] is warm_p and call[3][1] is warm_blocks


def test_checkpoint_resume_reproduces_uninterrupted_fit_and_rejects_tampering():
    x, _ = sequences(304, count=2, steps=60, sigma=2.0)
    full, _, full_history = fit_adaptive(x, 0.01, iterations=7)
    saved = {}

    def interrupted(p, blocks, history):
        saved.update(p=p, blocks=blocks, history=history)
        if history[-1]["completed_updates"] == 5:
            raise RuntimeError("simulated runtime loss")

    with pytest.raises(RuntimeError, match="runtime loss"):
        fit_adaptive(x, 0.01, iterations=7, checkpoint=interrupted)
    resumed, _, history = fit_adaptive(
        x,
        0.01,
        iterations=7,
        initial_state=(saved["p"], saved["blocks"]),
        resume_history=saved["history"],
    )
    np.testing.assert_array_equal(resumed, full)
    np.testing.assert_allclose([h["ll"] for h in history], [h["ll"] for h in full_history])
    bad = [dict(h) for h in saved["history"]]
    bad[-1]["ll"] += 1
    with pytest.raises(ValueError, match="reproduce"):
        fit_adaptive(
            x, 0.01, iterations=7, initial_state=(saved["p"], saved["blocks"]), resume_history=bad
        )


def test_sample_checkpoint_requires_matching_metadata_and_order(tmp_path):
    import copy
    import json

    from scripts.compare_adaptive_sample_size import load_checkpoint, save

    expected = {
        "schema": "test",
        "execution": {"source": "a"},
        "contract": {},
        "runs": [],
        "active": None,
    }
    path = tmp_path / "out.json"
    save(path, expected)
    assert load_checkpoint(path, expected) == expected
    bad = copy.deepcopy(expected)
    bad["execution"]["source"] = "b"
    save(path, bad)
    with pytest.raises(ValueError, match="execution"):
        load_checkpoint(path, expected)
    bad = copy.deepcopy(expected)
    bad["runs"] = [{"train_count": 1024}]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="prefix"):
        load_checkpoint(path, expected)


@pytest.mark.parametrize("large_only", [False, True])
def test_sample_cli_resumes_interrupted_large_phase_and_skips_completed_arm(
    monkeypatch, tmp_path, large_only
):
    import json
    import sys

    import scripts.compare_adaptive_sample_size as sample

    def data(seed, *, count, **kwargs):
        return np.full((count, 2, 1), seed, dtype=float), np.zeros((count, 2), dtype=int)

    calls = []
    fail_once = [True]

    def fit(
        x, penalty, *, iterations, tag, initial_state=None, resume_history=None, checkpoint=None
    ):
        calls.append((len(x), tag, len(resume_history or [])))
        p = np.full((4, 260), 1 / 260)
        blocks = [[(0, 260)]] * 4
        first = len(resume_history) - 1 if resume_history else 0
        history = list(resume_history[:-1]) if resume_history else []
        for update in range(first, iterations + 1):
            history.append({"completed_updates": update, "ll": 0.0})
            if update == first or update % 5 == 0 or update == iterations:
                checkpoint(p, blocks, history)
                if len(x) == 1024 and "adaptive" in tag and update == 5 and fail_once[0]:
                    fail_once[0] = False
                    raise RuntimeError("interrupted large fit")
        return p, blocks, history

    monkeypatch.setattr(sample, "sequences", data)
    monkeypatch.setattr(sample, "fit_adaptive", fit)
    monkeypatch.setattr(sample, "aggregate", lambda scores, p: (0.0, None))
    monkeypatch.setattr(sample, "metrics", lambda p, truth: {"fine_pmf_gate": False})
    monkeypatch.setattr(sample, "posterior_gradient_check", lambda *a, **k: {})
    monkeypatch.setattr(sample, "public_check", lambda *a, **k: {})
    path = tmp_path / "sample.json"
    argv = ["sample", "--base-commit", "test", "--output", str(path)]
    if large_only:
        argv += ["--large-only", "--seed", "1904"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="large fit"):
        sample.main()
    saved = json.loads(path.read_text())
    assert [r["train_count"] for r in saved["runs"]] == ([] if large_only else [128])
    assert saved["contract"]["seed"] == (1904 if large_only else 1903)
    assert saved["active"]["phase"] == "fit" and len(saved["active"]["history"]) == 6
    before = len(calls)
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    sample.main()
    assert calls[before:] == [(1024, "count=1024 adaptive", 6)]
    final = json.loads(path.read_text())
    assert [r["train_count"] for r in final["runs"]] == (
        [1024] if large_only else [128, 1024]
    ) and final["active"] is None
    before = len(calls)
    sample.main()
    assert len(calls) == before


def test_large_only_checkpoint_cannot_resume_other_seed_or_small_arm(tmp_path):
    import copy

    from scripts.compare_adaptive_sample_size import load_checkpoint, save

    expected = {
        "schema": "test",
        "execution": {},
        "contract": {"seed": 1904, "train_counts": [1024]},
        "runs": [],
        "active": None,
    }
    path = tmp_path / "large.json"
    save(path, expected)
    other = copy.deepcopy(expected)
    other["contract"]["seed"] = 1905
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, other)
    bad = copy.deepcopy(expected)
    bad["active"] = {"train_count": 128}
    save(path, bad)
    with pytest.raises(ValueError, match="active arm"):
        load_checkpoint(path, expected)
