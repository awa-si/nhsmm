from __future__ import annotations

import numpy as np

from scripts.generate_trading_regime_dataset import BASE_TRANSITION, REGIMES, generate


def test_trading_regime_dataset_is_deterministic_and_well_formed() -> None:
    data_a, meta_a = generate(rows=3000, seed=777, start_price=10_000.0)
    data_b, meta_b = generate(rows=3000, seed=777, start_price=10_000.0)

    assert meta_a["schema"] == "nhsmm-synthetic-trading-regimes-v1"
    assert meta_a["regime_row_counts"] == meta_b["regime_row_counts"]
    assert set(meta_a["regime_row_counts"]) == set(REGIMES)
    assert sum(meta_a["regime_row_counts"].values()) == 3000

    for key in data_a:
        np.testing.assert_allclose(data_a[key], data_b[key])
        assert len(data_a[key]) == 3000

    assert np.all(data_a["high"] >= np.maximum(data_a["open"], data_a["close"]))
    assert np.all(data_a["low"] <= np.minimum(data_a["open"], data_a["close"]))
    assert np.all(data_a["volume"] > 0.0)
    assert np.all(data_a["gt_episode_age"] >= 1)
    assert np.all(data_a["gt_remaining_duration"] >= 0)


def test_ground_truth_transition_probabilities_are_normalized_and_nonself() -> None:
    data, _ = generate(rows=3000, seed=778, start_price=10_000.0)
    probabilities = np.column_stack([data[f"gt_p_next_{name}"] for name in REGIMES])

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    current = data["gt_regime_id"]
    assert np.all(probabilities[np.arange(len(current)), current] == 0.0)
    np.testing.assert_allclose(BASE_TRANSITION.sum(axis=1), 1.0)


def test_dataset_contains_distinct_regime_signatures() -> None:
    data, _ = generate(rows=8000, seed=779, start_price=10_000.0)
    state = data["gt_regime_id"]
    ret = data["log_return_1"]

    means = {name: float(np.mean(ret[state == i])) for i, name in enumerate(REGIMES)}
    vols = {name: float(np.std(ret[state == i])) for i, name in enumerate(REGIMES)}

    assert means["bull"] > 0.0
    assert means["bear"] < 0.0
    assert vols["chaotic"] > vols["bull"]
    assert vols["chaotic"] > vols["bear"]
    assert vols["range"] < vols["bull"]
