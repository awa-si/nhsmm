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
