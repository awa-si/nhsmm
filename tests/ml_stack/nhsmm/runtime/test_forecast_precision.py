"""Independent rare-event and hard-support duration/transition references."""

from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
import torch
from nhsmm.filtering import HSMMFilterState, duration_log_hazard
from nhsmm.survival import active_episode_survival_forecast
from nhsmm.transitions import one_step_transition_forecast


@pytest.mark.parametrize("dtype,risk", [(torch.float32, 1e-9), (torch.float64, 1e-18)])
@pytest.mark.parametrize("tail", [False, True])
def test_small_duration_end_risk_survives_roundoff(dtype, risk, tail):
    D = 1 if tail else 2
    posterior = torch.full((1, 1, D), -torch.inf, dtype=dtype)
    posterior[..., 0] = 0
    state = HSMMFilterState(posterior)
    hazard = torch.tensor([risk], dtype=dtype)
    scores = torch.zeros((1, 1, D), dtype=dtype)
    if not tail:
        scores[..., 0] = hazard.log()
    horizons = [1, 3, 100_000] if tail else [1]
    forecast = active_episode_survival_forecast(state, scores, horizons, hazard if tail else None)
    with localcontext() as ctx:
        ctx.prec = 80
        p = Decimal.from_float(float(hazard[0]))
        if not tail:
            # Scores [log(p), 0] imply mass p/(1+p).
            p = p / (1 + p)
        expected = [float(1 - (1 - p) ** h) for h in horizons]
    torch.testing.assert_close(
        forecast.end_within_probability,
        torch.tensor([expected], dtype=dtype),
        rtol=2e-5 if dtype == torch.float32 else 1e-12,
        atol=0,
    )
    _, log_continue = duration_log_hazard(scores, hazard if tail else None)
    assert bool((log_continue[..., 0] < 0).all())


@pytest.mark.parametrize("dtype,risk", [(torch.float32, 1e-9), (torch.float64, 1e-18)])
def test_rare_state_change_is_off_diagonal_mass(dtype, risk):
    state = HSMMFilterState(torch.tensor([[[0.0], [-torch.inf]]], dtype=dtype))
    scores = torch.zeros(1, 2, 1, 2, dtype=dtype)
    scores[0, 0, 0, 1] = torch.tensor(risk, dtype=dtype).log()
    forecast = one_step_transition_forecast(state, torch.zeros(1, 2, 1, dtype=dtype), scores)
    with localcontext() as ctx:
        ctx.prec = 80
        odds = Decimal.from_float(float(scores[0, 0, 0, 1])).exp()
        expected = float(odds / (1 + odds))
    torch.testing.assert_close(
        forecast.state_change_probability,
        torch.tensor([expected], dtype=dtype),
        rtol=2e-6 if dtype == torch.float32 else 1e-12,
        atol=0,
    )


@pytest.mark.parametrize("forecast_kind", ["transition", "survival"])
def test_hard_support_forecast_has_finite_gradients(forecast_kind):
    logits = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    posterior = torch.stack(
        [logits.log_softmax(0), torch.full_like(logits, -torch.inf)], dim=-1
    ).unsqueeze(0)
    state = HSMMFilterState(posterior)
    duration = torch.tensor([[[0.4, -0.2], [0.7, 0.1]]], dtype=torch.float64, requires_grad=True)
    if forecast_kind == "survival":
        forecast = active_episode_survival_forecast(state, duration, [1, 2, 5])
        loss = forecast.end_within_probability.sum()
        expected = (logits.softmax(0) * duration.softmax(-1)[0, :, 0]).sum() + 2
    else:
        # Exact structural zeros and an unreachable destination create all--inf reductions.
        transition = torch.tensor(
            [[[[0.0, -torch.inf], [0.0, -torch.inf]], [[0.0, -torch.inf], [0.0, -torch.inf]]]],
            dtype=torch.float64,
        )
        forecast = one_step_transition_forecast(state, duration, transition)
        loss = forecast.next_state_prior[0, 0] + forecast.episode_end_probability.sum()
        p, q = logits.softmax(0), duration.softmax(-1)[0, :, 0]
        expected = p[0] + p[1] * q[1] + (p * q).sum()
    actual_grad = torch.autograd.grad(loss, (logits, duration), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (logits, duration))
    for actual, reference in zip(actual_grad, expected_grad):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_transition_result_rejects_inconsistent_state_change():
    state = HSMMFilterState(torch.tensor([[[0.0], [-torch.inf]]], dtype=torch.float64))
    transition = torch.tensor([[[[0.0, -torch.inf]], [[-torch.inf, 0.0]]]], dtype=torch.float64)
    forecast = one_step_transition_forecast(
        state, torch.zeros(1, 2, 1, dtype=torch.float64), transition
    )
    with pytest.raises(ValueError, match="off-diagonal"):
        replace(forecast, state_change_probability=torch.tensor([0.2], dtype=torch.float64))


@pytest.mark.parametrize("risk", [1e-9, 0.2, 0.8])
def test_tail_end_forecast_gradient_matches_geometric_reference(risk):
    state = HSMMFilterState(torch.zeros(1, 1, 1, dtype=torch.float64))
    tail = torch.tensor([risk], dtype=torch.float64, requires_grad=True)
    horizons = [1, 3, 7]
    result = active_episode_survival_forecast(
        state, torch.zeros(1, 1, 1, dtype=torch.float64), horizons, tail
    )
    gradient = torch.autograd.grad(result.end_within_probability.sum(), tail, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.sum(), tail)[0]
    expected_gradient = sum(h * (1 - risk) ** (h - 1) for h in horizons)
    expected_hessian = -sum(h * (h - 1) * (1 - risk) ** (h - 2) for h in horizons if h > 1)
    torch.testing.assert_close(
        gradient, torch.tensor([expected_gradient], dtype=tail.dtype), atol=0, rtol=1e-12
    )
    torch.testing.assert_close(
        hessian, torch.tensor([expected_hessian], dtype=tail.dtype), atol=0, rtol=1e-12
    )


def test_duration_and_transition_heads_learn_distinct_boundary_targets():
    from nhsmm.distributions.default import Duration, Transition

    torch.manual_seed(51)
    duration = Duration(2, "tanh", 2, init_mode="uniform").double()
    transition = Transition(2, 1, "tanh", "ergodic", init_mode="uniform", max_duration=2).double()
    posterior = torch.full((2, 2, 2), -torch.inf, dtype=torch.float64)
    posterior[0, 0, 0] = posterior[1, 1, 0] = 0
    state = HSMMFilterState(posterior)
    target_end = torch.tensor([0.2, 0.7], dtype=torch.float64)
    target_change = torch.tensor([0.15, 0.35], dtype=torch.float64)
    optimizer = torch.optim.Adam([duration.logits, transition.logits], lr=0.08)
    for _ in range(100):
        optimizer.zero_grad()
        result = one_step_transition_forecast(
            state,
            duration.log_matrix()[0].expand(2, -1, -1),
            transition.log_matrix()[0].expand(2, -1, -1, -1),
        )
        loss = (result.episode_end_probability - target_end).square().sum() + (
            result.state_change_probability - target_change
        ).square().sum()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(duration.logits.grad).all()
        assert torch.isfinite(transition.logits.grad).all()
        optimizer.step()
    torch.testing.assert_close(result.episode_end_probability, target_end, atol=0.005, rtol=0)
    torch.testing.assert_close(result.state_change_probability, target_change, atol=0.005, rtol=0)
