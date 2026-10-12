"""Acceptance contract for migrating neural scores to predictive likelihoods.

Strict expected failures identify the current production gaps. Remove each mark
only with the corresponding verified production migration.
"""

import math
import types

import pytest
import torch
from torch import nn

from nhsmm import HSMMFilterRuntime, ModelConfig, NHSMM, build_artifact, load_artifact

PENDING = pytest.mark.xfail(
    strict=True, reason="Step 1.1: predictive neural likelihood production migration pending"
)


class IdentityEncoder(nn.Module):
    causal = True

    def forward(self, x, mask=None):
        return x


def identity_model():
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=True,
            context_dim=1,
            dropout=0,
            seed=4,
        ),
        encoder=IdentityEncoder(),
        device="cpu",
    )
    model.initialize_distributions()
    model.eval()
    return model


def capture_emission_context(model):
    seen = []
    original = model.dist.emission._modulate

    def capture(self, *args, **kwargs):
        context = kwargs.get("context", args[0] if args else None)
        seen.append(None if context is None else context.detach().clone())
        return original(*args, **kwargs)

    model.dist.emission._modulate = types.MethodType(capture, model.dist.emission)
    return seen


@PENDING
def test_internal_emission_context_is_strictly_past_with_zero_start():
    model = identity_model()
    seen = capture_emission_context(model)
    x = torch.tensor([[[0.2], [0.4], [0.8]]])
    model._build_sequence_set(x)
    torch.testing.assert_close(seen[-1], torch.tensor([[[0.0], [0.2], [0.4]]]))


@PENDING
def test_internal_initial_context_does_not_depend_on_first_observation():
    model = identity_model()
    positive = model._build_sequence_set(torch.tensor([[[0.7]]])).canonical
    negative = model._build_sequence_set(torch.tensor([[[-0.9]]])).canonical
    torch.testing.assert_close(positive, negative)
    torch.testing.assert_close(positive, torch.zeros_like(positive))


@PENDING
def test_incremental_runtime_first_emission_uses_zero_context():
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=True,
            use_context_encoder=True,
            context_dim=2,
            cnn_channels=2,
            dropout=0,
            seed=11,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.eval()
    seen = capture_emission_context(model)
    HSMMFilterRuntime(model).step(torch.tensor([0.7]))
    torch.testing.assert_close(seen[-1], torch.zeros_like(seen[-1]), atol=1e-8, rtol=0)


def density_fixture():
    model = identity_model()
    with torch.no_grad():
        for p in model.dist.parameters():
            p.zero_()
        emission = model.dist.emission
        emission.mu[1, 0] = 4.0
        emission.log_var.fill_(math.log(math.expm1(1.0)))
        emission.delta_scale.fill_(0.5)
        emission._proj.weight.fill_(1.0)
        emission.context_net[0].weight[0, 0] = 0.001
        emission.context_net[0].weight[1, 0] = -0.001
        emission.context_net[1].weight.fill_(1.0)
        emission.context_net[3].weight[0, 0] = 1.0
        emission.context_net[3].weight[1, 0] = -1.0
    return model


@PENDING
def test_one_row_neural_likelihood_integrates_to_one():
    model = density_fixture()
    grid = torch.linspace(-12, 16, 28001)
    with torch.inference_mode():
        scores = model.log_likelihood(grid[:, None, None])
    integral = float(torch.trapezoid(scores.exp(), grid))
    assert integral == pytest.approx(1.0, abs=2e-5)


def test_exogenous_fixed_context_density_is_normalized_control():
    model = density_fixture()
    grid = torch.linspace(-12, 16, 28001)
    with torch.inference_mode():
        scores = model.log_likelihood(grid[:, None, None], context=torch.zeros(len(grid), 1, 1))
    assert float(torch.trapezoid(scores.exp(), grid)) == pytest.approx(1.0, abs=2e-5)


@PENDING
def test_neural_predictive_density_integrates_to_one_after_fixed_prefix():
    model = density_fixture()
    grid = torch.linspace(-12, 16, 28001)
    prefix = torch.full((1, 1, 1), 0.7)
    candidates = torch.cat([prefix.expand(len(grid), -1, -1), grid[:, None, None]], dim=1)
    with torch.inference_mode():
        conditional = model.log_likelihood(candidates) - model.log_likelihood(prefix)[0]
    assert float(torch.trapezoid(conditional.exp(), grid)) == pytest.approx(1.0, abs=2e-5)


@PENDING
def test_legacy_encoder_artifact_requires_explicit_migration(tmp_path):
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=True,
            use_context_encoder=True,
            context_dim=2,
            cnn_channels=2,
            dropout=0,
            seed=11,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    payload = build_artifact(model)
    payload["artifact_version"] = 1
    path = tmp_path / "legacy-encoder.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="legacy|retrain|migration|timing"):
        load_artifact(path)


@PENDING
def test_noncausal_internal_encoder_cannot_claim_generative_likelihood():
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=False,
            use_context_encoder=True,
            context_dim=2,
            cnn_channels=2,
            dropout=0,
            seed=11,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.eval()
    with pytest.raises(ValueError, match="predictive|causal|generative"):
        model.log_likelihood(torch.tensor([[[0.2], [0.7]]]))
