import numpy as np
import pytest

from scripts.diagnose_trading_decoder_duration import (
    complete_episode_medians,
    episode_median,
)


@pytest.mark.parametrize(
    "p,h",
    [
        ([0.5, 0.5], 0.5),
        ([0.1, 0.1, 0.8], 0.2),
        ([1.0], 0.1),
        ([0.2, 0.8], 1.0),
        ([0.1, 0.2, 0.7], 0.5),
    ],
)
def test_episode_median_matches_explicit_geometric_masses(p, h):
    masses = list(p[:-1])
    masses.extend(p[-1] * h * (1 - h) ** i for i in range(1000))
    expected = int(np.flatnonzero(np.cumsum(masses) >= 0.5)[0] + 1)
    assert episode_median(p, h) == expected


def test_complete_episodes_exclude_censored_edges_and_include_self_resets():
    # Left partial episode; two complete same-state episodes; right partial.
    states = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 2]])
    ages = np.array([[4, 5, 1, 2, 3, 1, 2, 3, 1]])
    result = complete_episode_medians(states, ages)
    assert result["bear"] == {"count": 2, "median": 3.0}
    assert result["bull"] == {"count": 0, "median": None}
    assert result["range"] == {"count": 0, "median": None}


@pytest.mark.parametrize("p,h", [([0.2, 0.2], 0.5), ([-0.2, 1.2], 0.5), ([1.0], 0)])
def test_episode_median_rejects_invalid_distributions(p, h):
    with pytest.raises(ValueError):
        episode_median(p, h)
