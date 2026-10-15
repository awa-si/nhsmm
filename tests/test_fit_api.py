"""Fit is the initialized optimize path, with exact math and state parity."""

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


@pytest.mark.parametrize("mode", ["dense", "ragged", "context", "encoder", "noncausal", "warm"])
def test_fit_matches_explicit_initialization_and_optimize(mode):
    config = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=3,
        causal=mode != "noncausal",
        seed=71,
        n_init=2,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        optimizer_temperature_mode="constant",
        use_context_encoder=mode == "encoder",
        context_dim=2 if mode == "context" else None,
    )
    x = torch.randn(2, 6, 2, generator=torch.Generator().manual_seed(81))
    context = x / 7 if mode == "context" else None
    if mode == "ragged":
        x = [x[0, :4], x[1]]
    reference = NHSMM(config, device="cpu")
    reference.initialize_distributions()
    reference.optimize(x, context=context)
    candidate = NHSMM(config, device="cpu")
    if mode == "warm":
        candidate.initialize_distributions()
        original = candidate.dist
    assert candidate.fit(x, context=context) is candidate
    if mode == "warm":
        # optimize owns restart replacement; fit must not perform an extra reset.
        assert type(candidate.dist) is type(original)
    assert candidate.optimization_summary() == reference.optimization_summary()
    assert candidate.training == reference.training
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate.log_likelihood(x, context=context),
        reference.log_likelihood(x, context=context),
        rtol=0,
        atol=0,
    )


def test_fit_preserves_initialized_custom_container(monkeypatch):
    config = ModelConfig(n_states=2, n_features=2, max_duration=3, causal=True, seed=31)
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    original = model.dist

    def optimize(x, context=None, cfg=None):
        assert model.dist is original
        assert cfg is config
        return model

    def no_reinitialization(*args, **kwargs):
        raise AssertionError("fit must retain caller-initialized distributions")

    monkeypatch.setattr(model, "initialize_distributions", no_reinitialization)
    monkeypatch.setattr(model, "optimize", optimize)
    assert model.fit(torch.zeros(1, 3, 2), cfg=config) is model


def test_failed_fit_propagates_without_success_summary():
    model = NHSMM(ModelConfig(n_states=2, n_features=2, max_duration=3, seed=31), device="cpu")
    with pytest.raises(ValueError, match="finite"):
        model.fit(torch.full((1, 3, 2), float("nan")))
    with pytest.raises(RuntimeError):
        model.optimization_summary()
