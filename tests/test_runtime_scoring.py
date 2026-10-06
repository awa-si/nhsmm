from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.runtime import _RuntimeScoreCache, _boundary_scores, _emission_log_prob


def _model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=97,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_runtime_score_cache_matches_reference_and_invalidates() -> None:
    model = _model()
    cache = _RuntimeScoreCache()
    context = torch.randn(2, 1, int(model.config.context_dim))
    observation = torch.randn(2, 1, model.config.n_features)

    with torch.inference_mode():
        duration_fast, transition_fast = _boundary_scores(
            model,
            context,
            temperature=None,
            cache=cache,
        )
        duration_ref, transition_ref = _boundary_scores(
            model,
            context,
            temperature=None,
            cache=None,
        )
        emission_fast = _emission_log_prob(model, observation, context, cache)
        emission_ref = _emission_log_prob(model, observation, context, None)

    assert torch.allclose(duration_fast, duration_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(transition_fast, transition_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(emission_fast, emission_ref, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        model.duration_logits_bias.add_(0.25)
        model.dist.emission.log_var.add_(0.1)

    with torch.inference_mode():
        duration_fast, transition_fast = _boundary_scores(
            model,
            context,
            temperature=None,
            cache=cache,
        )
        duration_ref, transition_ref = _boundary_scores(
            model,
            context,
            temperature=None,
            cache=None,
        )
        emission_fast = _emission_log_prob(model, observation, context, cache)
        emission_ref = _emission_log_prob(model, observation, context, None)

    assert torch.allclose(duration_fast, duration_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(transition_fast, transition_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(emission_fast, emission_ref, atol=1e-6, rtol=1e-6)


def test_runtime_score_cache_invalidates_on_parameter_replacement() -> None:
    model = _model()
    cache = _RuntimeScoreCache()
    context = torch.randn(2, 1, int(model.config.context_dim))
    observation = torch.randn(2, 1, model.config.n_features)

    with torch.inference_mode():
        _boundary_scores(model, context, temperature=None, cache=cache)
        _emission_log_prob(model, observation, context, cache)

    duration_delta = torch.zeros_like(model.duration_logits_bias)
    duration_delta[0, 0] = 2.0
    model.duration_logits_bias = torch.nn.Parameter(
        model.duration_logits_bias.detach().clone() + duration_delta
    )

    emission_delta = torch.zeros_like(model.dist.emission.log_var)
    emission_delta[0, 0] = 1.0
    replacement = torch.nn.Parameter(model.dist.emission.log_var.detach().clone() + emission_delta)
    with torch.no_grad():
        replacement.add_(0.0)
    model.dist.emission.log_var = replacement

    with torch.inference_mode():
        duration_fast, transition_fast = _boundary_scores(
            model,
            context,
            temperature=None,
            cache=cache,
        )
        duration_ref, transition_ref = _boundary_scores(
            model,
            context,
            temperature=None,
            cache=None,
        )
        emission_fast = _emission_log_prob(model, observation, context, cache)
        emission_ref = _emission_log_prob(model, observation, context, None)

    assert torch.allclose(duration_fast, duration_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(transition_fast, transition_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(emission_fast, emission_ref, atol=1e-6, rtol=1e-6)
