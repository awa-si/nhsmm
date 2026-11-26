"""Regression gates for the 2026-10-15 distribution audit."""

import pytest
import torch
from nhsmm.distributions.default import (
    Categorical,
    Duration,
    Emission,
    IndependentStudentT,
    Initial,
    Transition,
)
from nhsmm.filtering import duration_log_hazard


@pytest.mark.parametrize("parameterization", ["logits", "probs"])
def test_categorical_validation_occurs_after_parameter_initialization(parameterization):
    value = torch.tensor([0.0, 0.4, 0.6])
    distribution = Categorical(**{parameterization: value}, validate_args=True)
    assert distribution.batch_shape == torch.Size()
    torch.testing.assert_close(distribution.probs.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("value", [torch.tensor(1), torch.tensor([1]), torch.tensor([[1], [2]])])
def test_categorical_labels_broadcast_like_torch(value):
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    actual = Categorical(logits=logits).log_prob(value)
    expected = torch.distributions.Categorical(logits=logits).log_prob(value)
    torch.testing.assert_close(actual, expected)


def test_zero_probability_keeps_finite_observed_class_gradients():
    probabilities = torch.tensor([0.0, 0.4, 0.6], dtype=torch.float64, requires_grad=True)
    distribution = Categorical(probs=probabilities, validate_args=True)
    assert torch.isneginf(distribution.log_prob(torch.tensor(0)))
    distribution.log_prob(torch.tensor(1)).backward()
    torch.testing.assert_close(
        probabilities.grad, torch.tensor([-1.0, 1.5, -1.0], dtype=torch.float64)
    )


def test_categorical_all_impossible_rejected_but_partial_support_valid():
    with pytest.raises(ValueError, match="finite mass"):
        Categorical(logits=torch.full((2, 3), -torch.inf), validate_args=True)
    distribution = Categorical(logits=torch.tensor([-torch.inf, 0.0, 0.0]), validate_args=True)
    torch.testing.assert_close(distribution.probs, torch.tensor([0.0, 0.5, 0.5]))


@pytest.mark.parametrize("scalar_location", [False, True])
@pytest.mark.parametrize("event_dim", [0, 1, 2])
def test_studentt_broadcasts_all_statistics_like_torch(scalar_location, event_dim):
    loc = torch.tensor(0.0) if scalar_location else torch.zeros(2, 3)
    scale = torch.ones(2, 3) if scalar_location else torch.tensor(1.0)
    df = torch.tensor(5.0)
    actual = IndependentStudentT(loc, scale, df, event_dim=event_dim)
    expected = torch.distributions.Independent(
        torch.distributions.StudentT(df=df, loc=loc, scale=scale), event_dim
    )
    assert actual.batch_shape == expected.batch_shape
    assert actual.event_shape == expected.event_shape
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.variance, expected.variance)
    torch.testing.assert_close(actual.entropy(), expected.entropy())
    assert actual.sample((4,)).shape == expected.sample((4,)).shape
    torch.testing.assert_close(
        actual.log_prob(torch.zeros(2, 3)), expected.log_prob(torch.zeros(2, 3))
    )


@pytest.mark.parametrize("family", ["gaussian", "studentt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_emission_locations_preserve_large_translation_and_gradients(family, dtype):
    distribution = Emission(2, 1, "tanh", family).to(dtype=dtype)
    location = distribution.mu if family == "gaussian" else distribution.loc
    with torch.no_grad():
        location.copy_(torch.tensor([[0.0], [1.0]], dtype=dtype))
    baseline = distribution.log_prob(torch.tensor([[[0.0]]], dtype=dtype))
    baseline.sum().backward()
    expected_gradient = location.grad.clone()
    distribution.zero_grad()
    with torch.no_grad():
        location.add_(1e6)
    translated = distribution.log_prob(torch.tensor([[[1e6]]], dtype=dtype))
    translated.sum().backward()
    torch.testing.assert_close(translated, baseline)
    torch.testing.assert_close(location.grad, expected_gradient)
    assert location.grad.abs().sum() > 0


@pytest.mark.parametrize("component", ["initial", "duration", "transition"])
def test_hard_support_is_exact_even_for_saturated_finite_scores(component):
    if component == "initial":
        distribution = Initial(2, "tanh")
        mask = torch.tensor([True, False])
    elif component == "duration":
        distribution = Duration(1, "tanh", max_duration=2)
        mask = torch.tensor([True, False])
    else:
        distribution = Transition(2, 1, "tanh", "ergodic", max_duration=1)
        mask = torch.tensor([[[True, False]], [[False, True]]])
    distribution = distribution.double()
    with torch.no_grad():
        distribution.logits.fill_(-1e40)
    logp = distribution.log_matrix(mask=mask)
    expanded = mask.expand_as(logp)
    assert torch.isneginf(logp[~expanded]).all()
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones_like(logp[..., 0]))


@pytest.mark.parametrize(
    "scores",
    [
        [0.0, -torch.inf, -torch.inf],
        [-torch.inf, 0.0, -torch.inf],
        [0.0, -torch.inf, 1.0],
    ],
)
def test_exact_zero_support_hazards_have_finite_gradients(scores):
    log_scores = torch.tensor([[scores]], dtype=torch.float64, requires_grad=True)
    end, continuation = duration_log_hazard(log_scores)
    assert not torch.isnan(end).any()
    assert not torch.isnan(continuation).any()
    (end.exp().sum() + continuation.exp().sum()).backward()
    assert torch.isfinite(log_scores.grad).all()


def test_sparse_hazard_gradcheck_and_probability_reference():
    mask = torch.tensor([[[True, False, True, False]]])
    raw = torch.tensor([[[0.2, 0.5, -0.4, 0.1]]], dtype=torch.float64, requires_grad=True)

    def outputs(scores):
        end, continuation = duration_log_hazard(scores.masked_fill(~mask, -torch.inf))
        return torch.cat((end.exp().flatten(), continuation.exp().flatten()))

    assert torch.autograd.gradcheck(outputs, (raw,))
    p = raw.masked_fill(~mask, -torch.inf).softmax(-1)
    survival = p.flip(-1).cumsum(-1).flip(-1)
    expected_end = torch.where(survival > 0, p / survival.clamp_min(1e-300), 0.0)
    actual_end, _ = duration_log_hazard(raw.masked_fill(~mask, -torch.inf))
    torch.testing.assert_close(actual_end.exp(), expected_end)
