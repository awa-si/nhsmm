from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet


def _sequence(log_emission: torch.Tensor) -> SequenceSet:
    T, _ = log_emission.shape
    return SequenceSet(
        sequences=torch.zeros(1, T, 1, dtype=log_emission.dtype),
        lengths=torch.tensor([T]),
        masks=torch.ones(1, T, 1, dtype=torch.bool),
        contexts=torch.zeros(1, T, 0, dtype=log_emission.dtype),
        canonical=torch.zeros(1, 1, 0, dtype=log_emission.dtype),
        log_probs=log_emission.unsqueeze(0),
    )


def _score_objective(
    causal: bool,
    initial_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    emission_log_prob: torch.Tensor,
) -> torch.Tensor:
    K = initial_logits.shape[0]
    D = duration_logits.shape[-1]
    T = emission_log_prob.shape[0]
    model = NHSMM(
        ModelConfig(
            n_states=K,
            n_features=1,
            max_duration=D,
            causal=causal,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    sequence = _sequence(emission_log_prob)
    initial = initial_logits.log_softmax(-1)
    duration = duration_logits.log_softmax(-1)
    transition = transition_logits.log_softmax(-1)
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        alpha = model.forward(sequence)
    return torch.logsumexp(alpha[0, T - 1].flatten(), dim=0)


@pytest.mark.parametrize("causal", [False, True])
def test_likelihood_recurrence_gradcheck_all_score_groups(causal: bool) -> None:
    dtype = torch.float64
    K, D, T = 2, 2, 3
    g = torch.Generator().manual_seed(9100 + int(causal))
    args = (
        torch.randn(K, generator=g, dtype=dtype, requires_grad=True),
        torch.randn(T, K, D, generator=g, dtype=dtype, requires_grad=True),
        torch.randn(T, K, D, K, generator=g, dtype=dtype, requires_grad=True),
        torch.randn(T, K, generator=g, dtype=dtype, requires_grad=True) - 0.2,
    )
    assert torch.autograd.gradcheck(
        lambda *xs: _score_objective(causal, *xs),
        args,
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=False,
    )


@contextmanager
def _perturb_scalar(parameter: torch.nn.Parameter, flat_index: int, delta: float):
    flat = parameter.view(-1)
    with torch.no_grad():
        original = flat[flat_index].item()
        flat[flat_index] = original + delta
    try:
        yield
    finally:
        with torch.no_grad():
            flat[flat_index] = original


def _finite_difference(
    model: NHSMM, x: torch.Tensor, parameter: torch.nn.Parameter, index: int
) -> float:
    eps = 2e-5
    with _perturb_scalar(parameter, index, eps):
        plus = float(model.log_likelihood(x, reduce=True).detach())
    with _perturb_scalar(parameter, index, -eps):
        minus = float(model.log_likelihood(x, reduce=True).detach())
    return (plus - minus) / (2.0 * eps)


def _actual_model(causal: bool, emission_type: str = "gaussian") -> tuple[NHSMM, torch.Tensor]:
    torch.manual_seed(9200 + int(causal) + (10 if emission_type == "studentt" else 0))
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=causal,
            emission_type=emission_type,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model = model.to(dtype=torch.float64)
    model.train()
    with torch.no_grad():
        model.dist.initial.logits.copy_(torch.tensor([0.2, -0.3], dtype=torch.float64))
        model.dist.duration.logits.copy_(
            torch.tensor([[0.3, -0.1], [-0.2, 0.4]], dtype=torch.float64)
        )
        model.dist.transition.logits.copy_(
            torch.tensor(
                [
                    [[0.2, -0.1], [0.4, -0.2]],
                    [[-0.3, 0.5], [0.1, 0.3]],
                ],
                dtype=torch.float64,
            )
        )
        model.duration_logits_bias.zero_()
        if emission_type == "gaussian":
            model.dist.emission.mu.copy_(torch.tensor([[-0.4], [0.6]], dtype=torch.float64))
            model.dist.emission.log_var.copy_(torch.tensor([[-0.2], [0.1]], dtype=torch.float64))
        else:
            model.dist.emission.loc.copy_(torch.tensor([[-0.4], [0.6]], dtype=torch.float64))
            model.dist.emission.scale_param.copy_(torch.tensor([[0.3], [0.5]], dtype=torch.float64))
            model.dist.emission.dof.copy_(torch.tensor([4.0, 6.0], dtype=torch.float64))
    x = torch.tensor([[[-0.6], [0.1], [0.8], [0.4]]], dtype=torch.float64)
    return model, x


@pytest.mark.parametrize("causal", [False, True])
def test_actual_gaussian_parameter_gradients_match_finite_difference(causal: bool) -> None:
    model, x = _actual_model(causal, "gaussian")
    model.zero_grad(set_to_none=True)
    objective = model.log_likelihood(x, reduce=True)
    objective.backward()

    checks = [
        (model.dist.initial.logits, 0),
        (model.dist.duration.logits, 1),
        (model.dist.transition.logits, 3),
        (model.duration_logits_bias, 1),
        (model.dist.emission.mu, 0),
        (model.dist.emission.log_var, 1),
    ]
    for parameter, index in checks:
        assert parameter.grad is not None
        analytical = float(parameter.grad.view(-1)[index])
        numerical = _finite_difference(model, x, parameter, index)
        assert analytical == pytest.approx(numerical, abs=2e-5, rel=3e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_actual_studentt_parameter_gradients_match_finite_difference(causal: bool) -> None:
    model, x = _actual_model(causal, "studentt")
    model.zero_grad(set_to_none=True)
    objective = model.log_likelihood(x, reduce=True)
    objective.backward()

    checks = [
        (model.dist.emission.loc, 0),
        (model.dist.emission.scale_param, 1),
        (model.dist.emission.dof, 0),
    ]
    for parameter, index in checks:
        assert parameter.grad is not None
        analytical = float(parameter.grad.view(-1)[index])
        numerical = _finite_difference(model, x, parameter, index)
        assert analytical == pytest.approx(numerical, abs=3e-5, rel=5e-4)


def test_plain_hsmm_frozen_controls_have_no_gradients() -> None:
    model, x = _actual_model(True, "gaussian")
    model.zero_grad(set_to_none=True)
    model.log_likelihood(x, reduce=True).backward()

    for component in (model.dist.initial, model.dist.duration, model.dist.transition):
        assert not component.log_temperature.requires_grad
        assert component.log_temperature.grad is None
        assert not component.delta_scale.requires_grad
        assert component.delta_scale.grad is None
    assert not model.dist.emission.log_temperature.requires_grad
    assert model.dist.emission.log_temperature.grad is None
    assert not model.dist.emission.logits.requires_grad
    assert model.dist.emission.logits.grad is None
