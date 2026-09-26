from __future__ import annotations

import torch

from nhsmm.distributions.default import Categorical, Duration, Emission, Initial, Transition


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
    duration = Duration(n_states=2, activation="tanh", max_duration=4, context_dim=None, init_mode="uniform")
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
    duration = Duration(n_states=1, activation="tanh", max_duration=3, context_dim=None, init_mode="uniform")
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
        transition.logits.copy_(torch.tensor([
            [[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]],
            [[1.0, 3.0, 0.0], [1.0, 0.0, 3.0]],
            [[0.0, 1.0, 3.0], [3.0, 1.0, 0.0]],
        ]))

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
