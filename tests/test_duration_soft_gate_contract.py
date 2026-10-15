"""Duration soft-gate agreement across public probability helpers."""

import pytest
import torch

from nhsmm.config import EPS
from nhsmm.distributions.default import Duration


@pytest.mark.parametrize("state_gate", [False, True])
@pytest.mark.parametrize("context_kind", ["none", "sequence", "timestep", "nested"])
@pytest.mark.parametrize("temperature", [0.4, 2.0])
def test_soft_gate_helpers_match_existing_model_formula(state_gate, context_kind, temperature):
    torch.manual_seed(37)
    duration = Duration(2, "tanh", max_duration=4, context_dim=3).double()
    context = None
    if context_kind != "none":
        shape = (2, 3, 5, 3) if context_kind == "nested" else (2, 5, 3)
        context = torch.randn(*shape, dtype=torch.float64)
    timestep = 2 if context_kind == "timestep" else None
    gate = torch.tensor([-100.0, -1.0, 1.5, 4.0], dtype=torch.float64)
    if state_gate:
        gate = torch.stack((gate, gate.flip(0)))
    gate.requires_grad_()
    mask = torch.tensor([[True, False, True, True], [True, True, False, True]])
    kwargs = dict(context=context, temperature=temperature, mask=mask)
    temporal_kwargs = dict(kwargs, timestep=timestep)
    ungated = duration.expected_probs(**temporal_kwargs)
    # Independent multiplicative reference: temperature applies to scores only.
    weights = ungated * gate.sigmoid().clamp_min(EPS)
    expected = weights / weights.sum(-1, keepdim=True)
    actual = duration.expected_probs(**temporal_kwargs, soft_dmax=gate)
    assert actual.shape == ungated.shape
    torch.testing.assert_close(actual, expected)
    log_matrix = duration.log_matrix(**temporal_kwargs, soft_dmax=gate)
    matrix_expected = expected
    while matrix_expected.ndim < 4:
        matrix_expected = matrix_expected.unsqueeze(0)
    torch.testing.assert_close(log_matrix.exp(), matrix_expected)
    labels = expected.argmax(-1)
    torch.testing.assert_close(
        duration.log_prob(labels, **temporal_kwargs, soft_dmax=gate),
        expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1).log(),
    )
    # forward has no timestep argument; compare it on full-context inputs.
    if timestep is None:
        scores = duration(**kwargs, soft_dmax=gate)
        assert scores.shape == ungated.shape
        torch.testing.assert_close(scores.softmax(-1), expected)
        torch.testing.assert_close(
            duration(**kwargs, soft_dmax=gate, return_dist=True).probs, expected
        )
    actual_gradient = torch.autograd.grad(
        actual[..., 0].sum(), (gate, duration.logits), retain_graph=True
    )
    expected_gradient = torch.autograd.grad(
        expected[..., 0].sum(), (gate, duration.logits)
    )
    for actual_grad, expected_grad in zip(actual_gradient, expected_gradient):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("method", ["expected_probs", "forward", "log_matrix", "log_prob"])
@pytest.mark.parametrize("shape", [(), (3,), (3, 4), (1, 2, 4)])
def test_soft_gate_shape_validation_is_shared(method, shape):
    duration = Duration(2, "tanh", max_duration=4)
    kwargs = dict(soft_dmax=torch.zeros(shape))
    if method == "log_prob":
        kwargs["x"] = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="soft_dmax"):
        getattr(duration, method)(**kwargs)


def test_soft_gate_preserves_log_matrix_time_expansion_and_tail_hazard():
    duration = Duration(2, "tanh", max_duration=4, tail_enabled=True)
    gate = torch.tensor([-2.0, 0.0, 1.0, 3.0])
    hazard = duration.tail_probability().clone()
    expected = duration.log_matrix(soft_dmax=gate)
    actual = duration.log_matrix(soft_dmax=gate, T=5)
    assert actual.shape == (1, 5, 2, 4)
    torch.testing.assert_close(actual, expected.expand_as(actual))
    torch.testing.assert_close(duration.tail_probability(), hazard)
