from __future__ import annotations

import copy

import pytest
import torch

from nhsmm.distributions.default import Duration, Emission, Initial, Transition


def _clone_pair(factory, **kwargs):
    torch.manual_seed(123)
    a = factory(**kwargs)
    b = copy.deepcopy(a)
    return a, b


@pytest.mark.parametrize("factory,kwargs", [
    (Initial, dict(n_states=3, activation="tanh", init_mode="normal", context_dim=2)),
    (Duration, dict(n_states=3, activation="tanh", max_duration=4, init_mode="normal", context_dim=2)),
    (Transition, dict(n_states=3, n_features=2, activation="tanh", transition_type="ergodic",
                      max_duration=4, init_mode="normal", context_dim=2)),
])
def test_categorical_component_initialize_does_not_bake_context_into_base(factory, kwargs) -> None:
    no_ctx, with_ctx = _clone_pair(factory, **kwargs)
    context = torch.tensor([0.25, -0.5])

    torch.manual_seed(999)
    no_ctx.initialize(context=None, jitter=0.0)
    torch.manual_seed(999)
    with_ctx.initialize(context=context, jitter=0.0)

    torch.testing.assert_close(with_ctx.base, no_ctx.base)


@pytest.mark.parametrize("emission_type", ["gaussian", "studentt"])
def test_emission_initialize_does_not_bake_context_into_base(emission_type: str) -> None:
    no_ctx, with_ctx = _clone_pair(
        Emission,
        n_states=3,
        n_features=2,
        activation="tanh",
        emission_type=emission_type,
        context_dim=2,
    )
    context = torch.tensor([0.25, -0.5])

    torch.manual_seed(999)
    no_ctx.initialize(mode="spread", context=None, jitter=0.0)
    torch.manual_seed(999)
    with_ctx.initialize(mode="spread", context=context, jitter=0.0)

    torch.testing.assert_close(with_ctx.base, no_ctx.base)


@pytest.mark.parametrize("factory,kwargs", [
    (Initial, dict(n_states=3, activation="tanh", init_mode="normal", context_dim=2)),
    (Duration, dict(n_states=3, activation="tanh", max_duration=4, init_mode="normal", context_dim=2)),
    (Transition, dict(n_states=3, n_features=2, activation="tanh", transition_type="ergodic",
                      max_duration=4, init_mode="normal", context_dim=2)),
])
def test_initialize_context_is_applied_exactly_once(factory, kwargs) -> None:
    component, reference = _clone_pair(factory, **kwargs)
    context = torch.tensor([0.25, -0.5])

    torch.manual_seed(999)
    returned = component.initialize(context=context, jitter=0.0)
    torch.manual_seed(999)
    reference.initialize(context=None, jitter=0.0)
    expected = reference._get_dist(context=context)

    torch.testing.assert_close(returned.log_probs, expected.log_probs)


@pytest.mark.parametrize("emission_type", ["gaussian", "studentt"])
def test_emission_initialize_context_is_applied_exactly_once(emission_type: str) -> None:
    component, reference = _clone_pair(
        Emission,
        n_states=3,
        n_features=2,
        activation="tanh",
        emission_type=emission_type,
        context_dim=2,
    )
    context = torch.tensor([0.25, -0.5])

    torch.manual_seed(999)
    returned = component.initialize(mode="spread", context=context, jitter=0.0)
    torch.manual_seed(999)
    reference.initialize(mode="spread", context=None, jitter=0.0)
    expected = reference._get_dist(context=context)

    x = torch.tensor([[[0.1, -0.2]]])
    actual_lp = returned.log_prob(x[..., None, :].expand(-1, -1, 3, -1))
    expected_lp = expected.log_prob(x[..., None, :].expand(-1, -1, 3, -1))
    torch.testing.assert_close(actual_lp, expected_lp)
