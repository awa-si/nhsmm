"""Independent probability-space checks for causal edge evidence diagnostics."""

import numpy as np
import pytest
import torch

from nhsmm.filtering import HSMMFilterState
from nhsmm.transitions import one_step_transition_forecast
from scripts.diagnose_trading_boundary_evidence import match_switches, update_edge


def test_edge_update_matches_probability_space_enumeration():
    prior = np.array([[0.1, 0.2], [0.3, 0.4]])
    end = np.array([[0.3, 0.2], [0.6, 0.4]])
    trans = np.array([[[0.8, 0.2], [0.4, 0.6]], [[0.3, 0.7], [0.9, 0.1]]])
    likelihood = np.array([0.25, 2.0])
    boundary = np.zeros((2, 2))
    continuation = np.zeros(2)
    for src in range(2):
        for age in range(2):
            continuation[src] += prior[src, age] * (1 - end[src, age]) * likelihood[src]
            for dst in range(2):
                boundary[src, dst] += (
                    prior[src, age] * end[src, age] * trans[src, age, dst] * likelihood[dst]
                )
    normalizer = boundary.sum() + continuation.sum()
    forecast = one_step_transition_forecast(
        HSMMFilterState(torch.tensor(prior).log()[None]),
        torch.tensor([[0.3, 0.7], [0.6, 0.4]], dtype=torch.float64).log()[None],
        torch.tensor(trans).log()[None],
        torch.tensor([0.2, 0.4], dtype=torch.float64),
    )
    result = update_edge(forecast, torch.tensor(likelihood).log()[None])
    np.testing.assert_allclose(result["joint"][0], boundary / normalizer, atol=1e-14)
    np.testing.assert_allclose(
        result["current"][0], (boundary.sum(0) + continuation) / normalizer, atol=1e-14
    )
    np.testing.assert_allclose(
        result["previous"][0], (boundary.sum(1) + continuation) / normalizer, atol=1e-14
    )
    assert float(result["end"][0]) == pytest.approx(boundary.sum() / normalizer, abs=1e-14)
    assert float(result["change"][0]) == pytest.approx(
        (boundary.sum() - np.trace(boundary)) / normalizer, abs=1e-14
    )


def test_marginal_argmax_can_flip_without_a_latent_transition():
    state = HSMMFilterState(torch.tensor([[[0.51, 0.0], [0.49, 0.0]]], dtype=torch.float64).log())
    duration = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float64).log()
    transition = torch.full((1, 2, 2, 2), 0.5, dtype=torch.float64).log()
    forecast = one_step_transition_forecast(state, duration, transition)
    updated = update_edge(forecast, torch.tensor([[1.0, 2.0]], dtype=torch.float64).log())
    assert int(state.state_posterior.argmax(-1)) == 0
    assert int(updated["current"].argmax(-1)) == 1
    assert int(updated["previous"].argmax(-1)) == 1
    assert float(updated["end"][0]) == 0.0
    assert float(updated["change"][0]) == 0.0


def test_observation_can_support_an_unlikely_predicted_boundary():
    state = HSMMFilterState(torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64).log())
    duration = torch.tensor([[[0.01, 0.99], [0.01, 0.99]]], dtype=torch.float64).log()
    transition = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float64
    ).log()
    forecast = one_step_transition_forecast(state, duration, transition)
    assert float(forecast.state_change_probability[0]) == pytest.approx(0.01)
    updated = update_edge(forecast, torch.tensor([[1.0, 1000.0]], dtype=torch.float64).log())
    assert float(updated["change"][0]) == pytest.approx(10.0 / 10.99)
    extreme = update_edge(forecast, torch.tensor([[-1000.0, 1000.0]], dtype=torch.float64))
    assert torch.isfinite(extreme["current"]).all()
    assert float(extreme["change"][0]) == pytest.approx(1.0)


def test_self_transition_resets_episode_without_changing_state():
    state = HSMMFilterState(torch.zeros((1, 1, 1), dtype=torch.float64))
    forecast = one_step_transition_forecast(
        state,
        torch.zeros((1, 1, 1), dtype=torch.float64),
        torch.zeros((1, 1, 1, 1), dtype=torch.float64),
    )
    updated = update_edge(forecast, torch.zeros((1, 1), dtype=torch.float64))
    assert float(updated["end"][0]) == 1.0
    assert float(updated["change"][0]) == 0.0


def test_event_matching_maximizes_cardinality_without_window_start_boundary():
    truth = np.zeros((1, 15), dtype=int)
    truth[:, 5:9] = 1
    pred = np.zeros_like(truth)
    pred[:, 7:12] = 1
    matched = match_switches(truth, pred, tolerance=3)
    assert np.flatnonzero(matched[0]).tolist() == [7, 12]
    assert not matched[0, 0]
