"""Boundary checks for the zero-update frozen-artifact diagnostic."""

import numpy as np
import pytest

from scripts.diagnose_trading_gradient_flow import FEATURES, observations


def protocol():
    return {
        "features": list(FEATURES), "ranges_rows_half_open": {"train": [0, 36000]},
        "window_anchors": {"train": [0]}, "window_width": 480,
        "feature_scaler_train_only": {"mean": [0.] * 5, "std": [1.] * 5},
    }


def test_observation_loader_needs_no_ground_truth():
    data = {name: np.arange(480, dtype=np.float32) for name in FEATURES}
    result = observations(data, protocol(), "train", smoke=True)
    assert result.shape == (1, 32, 5)
    np.testing.assert_array_equal(result[:, :, 0], np.arange(32)[None, :])


def test_test_split_is_never_admitted():
    with pytest.raises(ValueError, match="only train and valid"):
        observations({}, {}, "test")


@pytest.mark.parametrize("anchors", [[-1], [35521], [0, 100], [480, 0]])
def test_observation_loader_rejects_invalid_windows(anchors):
    p = protocol()
    p["window_anchors"]["train"] = anchors
    with pytest.raises(ValueError, match="window"):
        observations({}, p, "train")


def test_observation_loader_rejects_nonfinite_scaler():
    p = protocol()
    p["feature_scaler_train_only"]["std"][0] = float("nan")
    data = {name: np.zeros(480) for name in FEATURES}
    with pytest.raises(ValueError, match="scale"):
        observations(data, p, "train")
