from __future__ import annotations

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


def test_categorical_sampling_contracts() -> None:
    torch.manual_seed(1)
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    dist = Categorical(logits=logits)

    samples = dist.sample(torch.Size([5]))
    assert samples.shape == (5, 1)

    soft = dist.rsample(torch.Size([4]), temperature=0.7)
    assert soft.shape == (4, 1, 3)
    assert torch.allclose(soft.sum(-1), torch.ones(4, 1), atol=1e-6, rtol=1e-6)
    soft.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    hard = dist.rsample(torch.Size([4]), temperature=0.7, hard=True)
    assert torch.all(hard.sum(-1) == 1)
    assert torch.all((hard == 0) | (hard == 1))


def test_initial_emits_expected_state_axis() -> None:
    initial = Initial(n_states=3, activation="tanh", context_dim=None, init_mode="uniform")
    with torch.no_grad():
        initial.logits.copy_(torch.tensor([0.0, 1.0, 2.0]))

    logits = initial.log_matrix()
    probs = torch.softmax(logits, dim=-1)
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=-1)

    assert logits.shape == (1, 1, 3)
    assert torch.allclose(probs[0, 0], expected)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1))


def test_duration_emits_normalized_duration_pmf_and_hard_support() -> None:
    duration = Duration(
        n_states=2, activation="tanh", max_duration=4, context_dim=None, init_mode="uniform"
    )
    with torch.no_grad():
        duration.logits.copy_(torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]))

    logp = duration.log_matrix(T=3)
    assert logp.shape == (1, 3, 2, 4)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(1, 3, 2), atol=1e-6)

    mask = torch.tensor([[True, True, False, False], [False, True, True, False]])
    masked = duration.log_matrix(mask=mask)
    probs = masked.exp()
    assert probs.shape == (1, 1, 2, 4)
    assert torch.all(probs[..., 0, 2:] == 0)
    assert probs[..., 1, 0].item() == 0
    assert probs[..., 1, 3].item() == 0
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1, 2), atol=1e-6)


def test_duration_soft_gate_changes_duration_distribution() -> None:
    duration = Duration(
        n_states=1, activation="tanh", max_duration=3, context_dim=None, init_mode="uniform"
    )
    with torch.no_grad():
        duration.logits.zero_()

    base = duration.log_matrix().exp()
    gated = duration.log_matrix(soft_dmax=torch.tensor([8.0, 0.0, -8.0])).exp()

    assert base.shape == (1, 1, 1, 3)
    assert gated.shape == (1, 1, 1, 3)
    assert gated[0, 0, 0, 0] > gated[0, 0, 0, 1] > gated[0, 0, 0, 2]
    assert not torch.allclose(base, gated)
    assert torch.allclose(gated.sum(-1), torch.ones(1, 1, 1), atol=1e-6)


def test_transition_emits_normalized_boundary_probabilities() -> None:
    transition = Transition(
        n_states=3,
        n_features=2,
        activation="tanh",
        transition_type="ergodic",
        max_duration=2,
        context_dim=None,
        init_mode="uniform",
    )
    with torch.no_grad():
        transition.logits.copy_(
            torch.tensor(
                [
                    [[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]],
                    [[1.0, 3.0, 0.0], [1.0, 0.0, 3.0]],
                    [[0.0, 1.0, 3.0], [3.0, 1.0, 0.0]],
                ]
            )
        )

    logp = transition.log_matrix(T=4)
    assert logp.shape == (1, 4, 3, 2, 3)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(1, 4, 3, 2), atol=1e-6)


def test_transition_structural_support_is_exact() -> None:
    semi = Transition(
        n_states=3,
        n_features=1,
        activation="tanh",
        transition_type="semi",
        max_duration=2,
        context_dim=None,
        init_mode="uniform",
    )
    probs = semi.log_matrix().exp()
    assert probs.shape == (1, 1, 3, 2, 3)
    assert torch.all(probs[..., 0, :, 2] == 0)
    assert torch.all(probs[..., 1, :, 0] == 0)
    assert torch.all(probs[..., 2, :, :2] == 0)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1, 3, 2), atol=1e-6)

    ltr = Transition(
        n_states=4,
        n_features=1,
        activation="tanh",
        transition_type="left-to-right",
        max_duration=None,
        context_dim=None,
        init_mode="uniform",
    )
    ltr_probs = ltr.log_matrix().exp()
    assert ltr_probs.shape == (1, 1, 4, 4)
    assert torch.all(ltr_probs[0, 0].tril(-1) == 0)
    assert torch.allclose(ltr_probs.sum(-1), torch.ones(1, 1, 4), atol=1e-6)


def test_external_transition_mask_is_exact() -> None:
    transition = Transition(
        n_states=2,
        n_features=1,
        activation="tanh",
        transition_type="ergodic",
        max_duration=None,
        context_dim=None,
        init_mode="uniform",
    )
    mask = torch.tensor([[True, False], [False, True]])
    probs = transition.log_matrix(mask=mask).exp()
    assert torch.allclose(probs, torch.eye(2).view(1, 1, 2, 2))


def test_gaussian_emission_matches_manual_log_prob() -> None:
    emission = Emission(
        n_states=2,
        n_features=2,
        activation="tanh",
        emission_type="gaussian",
        context_dim=None,
    )
    with torch.no_grad():
        emission.mu.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        emission.log_var.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))

    x = torch.tensor([[[0.25, -0.5], [1.0, 0.25]]])
    actual = emission.log_prob(x)
    dist = emission._get_dist(context=None)
    x_exp = x[..., None, :].expand(-1, -1, 2, -1)
    expected = dist.log_prob(x_exp)

    assert actual.shape == (1, 2, 2)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(actual).all()


def test_studentt_emission_constructs_and_emits_finite_log_prob() -> None:
    emission = Emission(
        n_states=2,
        n_features=2,
        activation="tanh",
        emission_type="studentt",
        context_dim=None,
    )
    x = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]])
    logp = emission.log_prob(x)

    assert logp.shape == (1, 2, 2)
    assert torch.isfinite(logp).all()


def test_context_modulation_emits_expected_time_shapes() -> None:
    torch.manual_seed(4)
    B, T, H = 2, 5, 3
    context = torch.randn(B, T, H)

    initial = Initial(n_states=3, activation="tanh", context_dim=H)
    duration = Duration(n_states=3, activation="tanh", max_duration=4, context_dim=H)
    transition = Transition(
        n_states=3,
        n_features=2,
        activation="tanh",
        transition_type="ergodic",
        max_duration=4,
        context_dim=H,
    )
    emission = Emission(
        n_states=3,
        n_features=2,
        activation="tanh",
        emission_type="gaussian",
        context_dim=H,
    )

    assert initial.log_matrix(context=context).shape == (B, T, 3)
    assert duration.log_matrix(context=context).shape == (B, T, 3, 4)
    assert transition.log_matrix(context=context).shape == (B, T, 3, 4, 3)

    x = torch.randn(B, T, 2)
    assert emission.log_prob(x, context=context).shape == (B, T, 3)


def test_emission_log_prob_rejects_non_finite_result() -> None:
    emission = Emission(
        n_states=2,
        n_features=1,
        activation="tanh",
        emission_type="gaussian",
        context_dim=None,
    )

    with pytest.raises(FloatingPointError, match="NaN/Inf in emission log-prob"):
        emission.log_prob(torch.tensor([[[float("nan")]]]))


def test_independent_student_t_respects_event_dim_contract() -> None:
    loc = torch.zeros(2, 3)
    scale = torch.ones(2, 3)
    df = torch.full((2, 3), 5.0)
    value = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]])

    dist = IndependentStudentT(loc=loc, scale=scale, df=df, event_dim=1)
    reference = torch.distributions.Independent(
        torch.distributions.StudentT(df=df, loc=loc, scale=scale),
        1,
    )

    assert dist.batch_shape == torch.Size([2])
    assert dist.event_shape == torch.Size([3])
    assert torch.allclose(dist.log_prob(value), reference.log_prob(value))
    assert torch.allclose(dist.entropy(), reference.entropy())

    scalar_events = IndependentStudentT(loc=loc, scale=scale, df=df, event_dim=0)
    assert scalar_events.batch_shape == torch.Size([2, 3])
    assert scalar_events.event_shape == torch.Size()
    assert scalar_events.log_prob(value).shape == (2, 3)


def test_emission_grad_scale_is_applied_once() -> None:
    torch.manual_seed(11)
    emission = Emission(
        n_states=3,
        n_features=2,
        activation="tanh",
        emission_type="gaussian",
        context_dim=2,
    )
    base = emission.base
    context = torch.randn(1, 4, 2)

    unscaled = emission._apply_context(base, context=context)
    scaled = emission._apply_context(base, context=context, grad_scale=0.5)

    assert torch.allclose(scaled, unscaled * 0.5, atol=1e-6, rtol=1e-6)


def test_categorical_rejects_non_last_category_axis() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        Categorical(logits=torch.zeros(2, 3), dim=0)


def test_gaussian_emission_forward_preserves_absolute_locations() -> None:
    emission = Emission(
        n_states=2,
        n_features=1,
        activation="tanh",
        emission_type="gaussian",
        context_dim=None,
    )
    with torch.no_grad():
        emission.mu.copy_(torch.tensor([[10.0], [12.0]]))
        emission.log_var.zero_()

    loc, covariance = emission.forward()

    assert loc.shape == (1, 1, 2, 1)
    assert covariance.shape == (1, 1, 2, 1, 1)
    assert torch.allclose(loc[0, 0, :, 0], torch.tensor([10.0, 12.0]))


def test_gaussian_emission_log_prob_is_translation_consistent() -> None:
    emission = Emission(
        n_states=2,
        n_features=1,
        activation="tanh",
        emission_type="gaussian",
        context_dim=None,
    )
    with torch.no_grad():
        emission.mu.copy_(torch.tensor([[10.0], [12.0]]))
        emission.log_var.zero_()

    x = torch.tensor([[[10.0]]])
    logp = emission.log_prob(x)

    assert logp[0, 0, 0] > logp[0, 0, 1]


def test_independent_student_t_moments_match_torch_contract() -> None:
    df = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
    loc = torch.arange(5.0)
    scale = torch.ones(5)

    dist = IndependentStudentT(loc=loc, scale=scale, df=df, event_dim=0)
    reference = torch.distributions.StudentT(df=df, loc=loc, scale=scale)

    torch.testing.assert_close(dist.mean, reference.mean, equal_nan=True)
    torch.testing.assert_close(dist.variance, reference.variance, equal_nan=True)


@pytest.mark.parametrize("factory", [Duration, Transition])
def test_context_aware_initialization_accepts_vector_context(factory) -> None:
    context = torch.tensor([0.25, -0.5])
    if factory is Duration:
        dist = factory(
            n_states=3,
            activation="tanh",
            max_duration=4,
            context_dim=2,
            init_mode="normal",
        )
    else:
        dist = factory(
            n_states=3,
            n_features=2,
            activation="tanh",
            transition_type="ergodic",
            max_duration=4,
            context_dim=2,
            init_mode="normal",
        )

    initialized = dist.initialize(context=context)
    assert torch.isfinite(dist.logits).all()
    assert initialized is not None


def test_constructor_applies_declared_initialization_mode() -> None:
    torch.manual_seed(123)
    initial = Initial(n_states=3, activation="tanh", init_mode="biased", context_dim=None)
    expected = torch.log(torch.tensor([0.8, 0.5, 0.2]) / 1.5)
    assert torch.allclose(initial.logits, expected, atol=1e-4, rtol=1e-4)


def test_categorical_log_prob_supports_sample_dimensions() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0], [1.5, -0.5, 0.25]])
    dist = Categorical(logits=logits)
    reference = torch.distributions.Categorical(logits=logits)
    torch.manual_seed(19)
    samples = reference.sample((5,))

    assert dist.log_prob(samples).shape == (5, 2)
    torch.testing.assert_close(dist.log_prob(samples), reference.log_prob(samples))


@pytest.mark.parametrize(
    "probs",
    [
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.5, -0.1, 0.6]]),
        torch.tensor([[0.5, float("nan"), 0.5]]),
    ],
)
def test_categorical_rejects_invalid_probabilities(probs) -> None:
    with pytest.raises(ValueError):
        Categorical(probs=probs)


def test_independent_student_t_expand_preserves_distribution_contract() -> None:
    base = IndependentStudentT(
        loc=torch.zeros(2, 3),
        scale=torch.ones(2, 3),
        df=torch.full((2, 3), 5.0),
        event_dim=1,
    )
    expanded = base.expand((4, 2))

    assert expanded.batch_shape == torch.Size([4, 2])
    assert expanded.event_shape == torch.Size([3])
    value = torch.zeros(4, 2, 3)
    assert expanded.log_prob(value).shape == (4, 2)


def test_initial_log_matrix_is_normalized_and_offset_invariant() -> None:
    initial = Initial(
        n_states=3,
        activation="tanh",
        context_dim=None,
        init_mode="normal",
    )
    with torch.no_grad():
        initial.logits.copy_(torch.tensor([-2.0, 0.5, 4.0]))
    baseline = initial.log_matrix()

    with torch.no_grad():
        initial.logits.add_(17.0)
    shifted = initial.log_matrix()

    torch.testing.assert_close(
        torch.logsumexp(baseline, dim=-1),
        torch.zeros_like(torch.logsumexp(baseline, dim=-1)),
    )
    torch.testing.assert_close(shifted, baseline)


def test_categorical_support_masks_reject_empty_rows() -> None:
    initial = Initial(n_states=2, activation="tanh", context_dim=None)
    with pytest.raises(ValueError, match="at least one state"):
        initial.log_matrix(mask=torch.tensor([False, False]))

    duration = Duration(
        n_states=2,
        activation="tanh",
        max_duration=3,
        context_dim=None,
    )
    with pytest.raises(ValueError, match="per state"):
        duration.log_matrix(mask=torch.tensor([[True, False, False], [False, False, False]]))

    transition = Transition(
        n_states=2,
        n_features=1,
        activation="tanh",
        transition_type="ergodic",
        max_duration=None,
        context_dim=None,
    )
    with pytest.raises(ValueError, match="at least one destination"):
        transition.log_matrix(mask=torch.tensor([[True, False], [False, False]]))


def test_emission_mask_broadcasts_over_batch_and_time() -> None:
    emission = Emission(
        n_states=2,
        n_features=2,
        activation="tanh",
        emission_type="gaussian",
        context_dim=3,
    )
    context = torch.randn(4, 5, 3)
    mask = torch.tensor([[True, False], [False, True]])
    constrained = emission._modulate(context=context, mask=mask)

    assert constrained.shape == (4, 5, 2, 2)
    assert torch.all(constrained[..., 0, 1] == 0)
    assert torch.all(constrained[..., 1, 0] == 0)


@pytest.mark.parametrize("emission_type", ["gaussian", "studentt"])
def test_emission_initialize_preserves_dtype(emission_type) -> None:
    emission = Emission(
        n_states=2,
        n_features=2,
        activation="tanh",
        emission_type=emission_type,
        context_dim=2,
    ).to(dtype=torch.float64)

    emission.initialize(context=torch.tensor([0.25, -0.5], dtype=torch.float64))

    for parameter in emission.parameters():
        assert parameter.dtype == torch.float64


@pytest.mark.parametrize("emission_type", ["gaussian", "studentt"])
def test_emission_temperature_does_not_rescale_locations(emission_type) -> None:
    emission = Emission(
        n_states=2,
        n_features=2,
        activation="tanh",
        emission_type=emission_type,
        context_dim=2,
    )
    context = torch.randn(3, 4, 2)

    baseline = emission._modulate(context=context, temperature=1.0)
    cooled = emission._modulate(context=context, temperature=0.25)

    torch.testing.assert_close(cooled, baseline)
    assert not emission.logits.requires_grad
    assert not emission.log_temperature.requires_grad


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([[0.0, float("nan"), 1.0]]),
        torch.tensor([[0.0, float("inf"), 1.0]]),
    ],
)
def test_categorical_rejects_nonfinite_logits(logits) -> None:
    with pytest.raises(ValueError, match="logits"):
        Categorical(logits=logits)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_categorical_rsample_rejects_invalid_temperature(temperature) -> None:
    dist = Categorical(logits=torch.zeros(2, 3))
    with pytest.raises(ValueError, match="temperature"):
        dist.rsample(temperature=temperature)


def test_categorical_rejects_nonfloating_inputs() -> None:
    with pytest.raises(TypeError, match="floating dtype"):
        Categorical(logits=torch.tensor([[1, 2, 3]]))


@pytest.mark.parametrize("factory", [Initial, Duration, Transition, Emission])
def test_distribution_constructors_reject_unknown_activation(factory) -> None:
    if factory is Initial:
        kwargs = dict(n_states=2, activation="bogus", context_dim=None)
    elif factory is Duration:
        kwargs = dict(n_states=2, activation="bogus", max_duration=3, context_dim=None)
    elif factory is Transition:
        kwargs = dict(
            n_states=2,
            n_features=1,
            activation="bogus",
            transition_type="ergodic",
            max_duration=2,
            context_dim=None,
        )
    else:
        kwargs = dict(
            n_states=2,
            n_features=1,
            activation="bogus",
            emission_type="gaussian",
            context_dim=None,
        )

    with pytest.raises(ValueError, match="Unsupported activation"):
        factory(**kwargs)
