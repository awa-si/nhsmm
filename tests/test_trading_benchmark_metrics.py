from __future__ import annotations

import itertools
import typing

import numpy as np
import pytest

from scripts import generate_trading_regime_dataset as generator
from scripts import validate_trading_regime_usefulness as benchmark


def _states(boundaries, length=7):
    result = np.zeros(length, dtype=int)
    for i, boundary in enumerate(boundaries):
        result[boundary:] = (i + 1) % 4
    return result[None, :]


def _maximum_matches(left, right, tolerance):
    # Independent dynamic programming reference for small boundary sets.
    table = np.zeros((len(left) + 1, len(right) + 1), dtype=int)
    for i, lvalue in enumerate(left, 1):
        for j, rvalue in enumerate(right, 1):
            table[i, j] = max(table[i - 1, j], table[i, j - 1])
            if abs(lvalue - rvalue) <= tolerance:
                table[i, j] = max(table[i, j], table[i - 1, j - 1] + 1)
    return int(table[-1, -1])


def test_boundary_matching_maximizes_cardinality():
    true = np.zeros((1, 16), dtype=int)
    true[:, 6:10] = 1
    true[:, 10:] = 2
    predicted = np.zeros((1, 16), dtype=int)
    predicted[:, 1:6] = 1
    predicted[:, 6:] = 2
    assert benchmark._boundary_f1(true, predicted, tol=5) == 1.0


def test_boundary_matching_matches_independent_reference():
    subsets = [
        tuple(i + 1 for i, include in enumerate(bits) if include)
        for bits in itertools.product((False, True), repeat=5)
    ]
    for left in subsets:
        for right in subsets:
            for tolerance in (0, 1, 2):
                count = _maximum_matches(left, right, tolerance)
                denominator = len(left) + len(right)
                expected = 2 * count / denominator if denominator else 1.0
                assert benchmark._boundary_f1(_states(left), _states(right), tolerance) == expected


def test_missing_state_cannot_pass_duration_acceptance():
    truth = {0: 4.0, 1: 6.0, 2: 8.0, 3: 10.0}
    prediction = benchmark._run_medians(np.array([[0, 0, 0, 0]]))
    error, missing = benchmark._duration_error(truth, prediction)
    assert error == 1.0
    assert missing == [1, 2, 3]
    row = {
        key: 0.0
        for key in (
            "matched_accuracy",
            "ari",
            "boundary_f1_tol5",
            "transition_mae",
            "full_ll_per_row",
            "hsmm_no_context_ll_per_row",
            "hmm_state_only_ll_per_row",
            "no_regime_gaussian_ll_per_row",
            "full_minus_hsmm_no_context",
            "full_minus_hmm",
            "full_minus_no_regime",
        )
    }
    row.update(duration_median_rel_error=0.0, duration_state_coverage=False)
    assert not benchmark._summary([row])["acceptance"]["duration_recovery"]


def test_generator_annotations_resolve():
    assert typing.get_type_hints(generator._duration)["rng"] is np.random.Generator
    hints = typing.get_type_hints(generator._next_state)
    assert hints["rng"] is np.random.Generator
    assert hints["return"] == tuple[int, np.ndarray]


def test_v2_transition_targets_use_last_observed_boundary_context():
    data, metadata = generator.generate(rows=3000, seed=777, start_price=10000)
    assert metadata["transition_context"]["conditioning_time"] == "episode_end"
    for episode in metadata["episodes"]:
        end = episode["end_row_exclusive"] - 1
        state = episode["regime_id"]
        _, expected = generator._next_state(
            np.random.default_rng(0), state, data["macro_stress"][end]
        )
        actual = np.array([data["gt_p_next_" + name][end] for name in generator.REGIMES])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_legacy_generation_preserves_frozen_numerical_reference():
    data, metadata = generator.generate(
        rows=3000, seed=424242, start_price=50000, transition_context_mode="episode_start"
    )
    assert metadata["schema"] == "nhsmm-synthetic-trading-regimes-v1"
    with np.load("data/reference/trading-regimes-v1/trading_regimes.npz") as frozen:
        for key in ("close", "macro_stress", "gt_regime_id", "gt_p_next_bull"):
            np.testing.assert_array_equal(data[key], frozen[key][:3000])


def test_budget_and_window_sampling_fail_fast():
    with pytest.raises(ValueError):
        benchmark.RunBudget(max_iter=0)
    with pytest.raises(ValueError):
        benchmark._windows({}, 0, 10, 1, window=20)
    with pytest.raises(ValueError):
        benchmark._boundary_f1(_states(()), _states(()), tol=-1)
