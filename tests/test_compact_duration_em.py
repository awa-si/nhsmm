"""Grouped duration Q optimizer, endpoint support and nested observation-only selection."""

import numpy as np
import pytest

from scripts.compare_compact_duration_em import (
    argument_parser,
    check_output_mode,
    compact_mstep,
    duration_support,
    fit_compact,
    group_score,
    partitions,
    public_check,
    select_width,
)
from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.verify_isolated_trading_hsmm import sequences, true_law


def test_support_defaults_full_and_oracle_is_explicit_and_retained(monkeypatch):
    import scripts.compare_compact_duration_em as compact

    args = argument_parser().parse_args(["--base-commit", "test", "--output", "out.json"])
    assert args.support_mode == "full"
    oracle = duration_support("oracle")
    np.testing.assert_array_equal(oracle, true_law() > 0)
    monkeypatch.setattr(compact, "true_law", lambda: pytest.fail("oracle accessed in full mode"))
    support = duration_support()
    assert support.shape == (4, 260) and support.all()
    with pytest.raises(ValueError):
        duration_support("invalid")


def test_full_support_public_and_gradient_checks_keep_probability_outside_oracle(monkeypatch):
    import scripts.compare_unlabelled_duration_em as reference

    support = duration_support()
    p = support / support.sum(1, keepdims=True)
    assert p[~duration_support("oracle")].sum() > 0
    x, labels = sequences(58, count=1, steps=80, sigma=2.0)
    monkeypatch.setattr(reference, "true_law", lambda: pytest.fail("hidden oracle gradient mask"))
    assert (
        reference.posterior_gradient_check(x, p, support=support)[
            "maximum_all_logit_gradient_error"
        ]
        < 1e-7
    )
    assert public_check(p, x, labels, support=support)["ll_error"] < 1e-7


def test_existing_oracle_evidence_cannot_be_overwritten_by_full_default(tmp_path):
    import json

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"contract": {}}))  # Historical v1 is oracle mode.
    check_output_mode(path, "oracle")
    with pytest.raises(ValueError, match="separate full path"):
        check_output_mode(path, "full")
    path.write_text(json.dumps({"contract": {"support_mode": "full"}}))
    check_output_mode(path, "full")
    with pytest.raises(ValueError):
        check_output_mode(path, "oracle")


def test_full_support_constrained_fit_is_finite_and_monotonic():
    x, _ = sequences(59, count=2, steps=80, sigma=2.0)
    p, history = fit_compact(x, duration_support(), 16, iterations=3)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
    assert np.isfinite(p).all()
    assert min(np.diff([h["ll"] for h in history])) >= -1e-6
    assert history[-1]["ll"] > history[0]["ll"]


def test_partition_covers_support_once_and_preserves_endpoint_atoms():
    support = np.array([[0, 1, 1, 1, 1, 1, 1, 0]], dtype=bool)
    groups = partitions(support, 3)[0]
    assert [g.tolist() for g in groups] == [[1], [2, 3, 4], [5], [6]]
    assert np.concatenate(groups).tolist() == list(range(1, 7))
    assert len(partitions(support, 1)[0]) == 6
    assert [len(g) for g in partitions(support, 100)[0]] == [1, 4, 1]
    with pytest.raises(ValueError):
        partitions(np.array([[1, 0, 1]], dtype=bool), 2)
    with pytest.raises(ValueError):
        partitions(support, 0)


def test_compact_mstep_matches_hand_solution_and_zero_group_score():
    c = np.array([[1.0, 3.0, 6.0, 10.0, 5.0]])
    groups = partitions(np.ones_like(c, dtype=bool), 3)
    p = compact_mstep(c, groups)
    np.testing.assert_allclose(p, [[1 / 25, 19 / 75, 19 / 75, 19 / 75, 5 / 25]], atol=1e-15)
    assert group_score(c, p, groups) < 1e-12
    # Perturb group masses within the same normalized family: exact solution maximizes Q.
    q = p.copy()
    q[0, 0] += 0.01
    q[0, -1] -= 0.01
    assert (c * np.log(p)).sum() > (c * np.log(q)).sum()
    with pytest.raises(ValueError):
        compact_mstep(np.array([[0.0, 0.0]]), partitions(np.ones((1, 2), dtype=bool), 1))


def test_unconstrained_width_one_recovers_expected_count_update():
    c = np.array([[0.0, 4.0, 2.0, 6.0, 0.0]])
    p = compact_mstep(c, partitions(c > 0, 1))
    np.testing.assert_allclose(p, c / c.sum(1, keepdims=True))


def test_constrained_em_monotonic_normalized_and_constant_inside_blocks():
    x, _ = sequences(55, count=2, steps=100, sigma=2.0)
    support = true_law() > 0
    p, history = fit_compact(x, support, 8, iterations=3)
    assert min(np.diff([r["ll"] for r in history])) >= -1e-6
    assert history[-1]["ll"] > history[0]["ll"]
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
    assert np.all(p[~support] == 0)
    for k, groups in enumerate(partitions(support, 8)):
        for g in groups:
            np.testing.assert_array_equal(p[k, g], np.full(len(g), p[k, g[0]]))


def test_group_score_matches_independent_marginal_finite_difference():
    x, _ = sequences(56, count=1, steps=100, sigma=2.0)
    support = true_law() > 0
    p = support / support.sum(1, keepdims=True)
    groups = partitions(support, 8)
    _, counts = aggregate(emission_scores(x), p)
    k, block = 3, groups[3][2]
    expected = counts[k, block].sum() - p[k, block].sum() * counts[k].sum()
    values = []
    for sign in (1, -1):
        q = p.copy()
        q[k, block] *= np.exp(sign * 1e-4)
        q[k] /= q[k].sum()
        values.append(aggregate(emission_scores(x), q)[0])
    assert expected == pytest.approx((values[0] - values[1]) / (2e-4), abs=1e-7)


def test_selection_is_disjoint_training_only_and_uses_validation_likelihood():
    x, _ = sequences(57, count=8, steps=60, sigma=2.0)
    width, report = select_width(x, true_law() > 0, seed=57, iterations=1, widths=(4, 8))
    assert set(report["train_indices"]).isdisjoint(report["validation_indices"])
    assert sorted(report["train_indices"] + report["validation_indices"]) == list(range(8))
    assert len(report["train_indices"]) == 6 and len(report["validation_indices"]) == 2
    assert (
        width == max(report["candidates"], key=lambda c: (c["validation_ll"], -c["width"]))["width"]
    )
