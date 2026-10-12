"""Independent clipped duration laws, label ledger and censoring controls."""

import math

import numpy as np
import pytest
import torch

from nhsmm.filtering import duration_log_hazard
from scripts.diagnose_trading_ground_truth import (
    audit_labels,
    duration_cdf,
    duration_pmf,
    geometric_projection,
    window_audit,
)
from scripts.generate_trading_regime_dataset import REGIMES, generate


def test_shifted_gamma2_cdf_matches_closed_form():
    x = np.array([0, 8, 21, 34, 105], dtype=float)
    z = np.maximum((x - 8) / 13, 0)
    expected = 1 - np.exp(-z) * (1 + z)
    np.testing.assert_allclose(duration_cdf("chaotic", x), expected, rtol=0, atol=1e-12)
    assert duration_cdf("bear", np.array([106.0]))[0] == pytest.approx(1 - math.exp(-1))


@pytest.mark.parametrize("regime", REGIMES)
def test_exact_duration_pmf_support_and_finite_hazard_match_core(regime):
    pmf = duration_pmf(regime)
    lower = {"bull": 42, "bear": 30, "range": 18, "chaotic": 8}[regime]
    assert np.all(pmf[: lower - 1] == 0)
    assert pmf[-1] > 0
    assert pmf.sum() == pytest.approx(1, abs=1e-12)
    hazard, _ = duration_log_hazard(torch.tensor(pmf, dtype=torch.float64).log()[None, None])
    np.testing.assert_allclose(
        hazard.exp().numpy()[0, 0], pmf / np.cumsum(pmf[::-1])[::-1], rtol=0, atol=1e-12
    )
    assert hazard.exp()[0, 0, -1] == 1


def test_geometric_projection_normalizes_and_matches_mean_with_positive_overflow():
    pmf = np.array([0.3, 0.21, 0.147, 0.1029, 0.07203, 0.16807])
    projection = geometric_projection(pmf, 2)
    h, mass = projection["tail_hazard"], projection["tail_mass"]
    ages = np.arange(1, 10001)
    approximation = np.r_[pmf[0], mass * h * (1 - h) ** np.arange(9999)]
    assert approximation.sum() == pytest.approx(1, abs=1e-12)
    assert approximation @ ages == pytest.approx(pmf @ np.arange(1, 7), abs=1e-12)
    assert projection["kl_nats_per_episode"] > 0
    assert projection["projection_tv"] > 0
    assert projection["probability_beyond_true_max"] > 0


def test_ground_truth_ledger_censoring_and_corruption_detection():
    data, metadata = generate(rows=3000, seed=779, start_price=10000)
    audit = audit_labels(data, metadata)
    assert audit["episodes"] == len(metadata["episodes"])
    if audit["terminal_right_censoring"]:
        terminal = audit["terminal_right_censoring"][0]
        assert terminal["stored_remaining_at_last_row"] == 0
        assert terminal["latent_remaining_after_last_row"] > 0
    copied = {k: v.copy() for k, v in data.items()}
    copied["gt_episode_age"][20] += 1
    with pytest.raises(AssertionError):
        audit_labels(copied, metadata)


def test_v2_exposes_latent_residual_life_and_censoring_without_changing_legacy_schema():
    data, metadata = generate(rows=3000, seed=779, start_price=10000)
    np.testing.assert_array_equal(
        data["gt_latent_remaining_duration"], data["gt_planned_duration"] - data["gt_episode_age"]
    )
    for episode in metadata["episodes"]:
        start, end = episode["start_row"], episode["end_row_exclusive"]
        np.testing.assert_array_equal(
            data["gt_episode_right_censored"][start:end], int(episode["right_censored"])
        )
        if episode["right_censored"]:
            assert data["gt_remaining_duration"][end - 1] == 0
            assert data["gt_latent_remaining_duration"][end - 1] > 0
    legacy, old = generate(
        rows=3000, seed=779, start_price=10000, transition_context_mode="episode_start"
    )
    assert "gt_latent_remaining_duration" not in legacy
    assert "gt_episode_right_censored" not in legacy
    assert "ground_truth_semantics" not in old
    assert all("right_censored" not in e for e in old["episodes"])


def test_window_audit_reports_mid_episode_initialization_and_incomplete_runs():
    data, metadata = generate(rows=3000, seed=779, start_price=10000)
    result = window_audit(data, metadata, 1, 241, 1, 240)
    assert result["left_censored_windows"] == 1
    assert result["windows"] == 1
    # No first partial episode can be counted as a complete episode.
    first = metadata["episodes"][0]
    expected = sum(
        e["start_row"] >= 1 and e["end_row_exclusive"] < 241 for e in metadata["episodes"]
    )
    assert sum(result["complete_episode_counts"].values()) == expected
    assert first["start_row"] == 0
