"""Supervised setup failures restore the caller's mode and trainability."""

import pytest
import torch
from nhsmm import NHSMM, ModelConfig

X = torch.tensor([[-1.0], [0.0], [1.0], [0.5]])
Z = torch.tensor([0, 0, 1, 1])
C = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]]])


def make(training):
    cfg = ModelConfig(
        2,
        1,
        max_duration=2,
        causal=True,
        allow_self_transitions=False,
        context_dim=2,
        seed=71,
        dropout=0.0,
        verbose=False,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    model.dist.emission.mu.requires_grad_(False)
    model.train(training)
    return model


def snapshot(model):
    return (
        {n: p.requires_grad for n, p in model.named_parameters()},
        {n: v.clone() for n, v in model.state_dict().items()},
        {n: m.training for n, m in model.named_modules()},
    )


def assert_restored(model, before):
    flags, state, modes = before
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert {n: m.training for n, m in model.named_modules()} == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("context_only", [True, False])
def test_optimizer_setup_failure_restores_flags_and_mode(monkeypatch, training, context_only):
    model = make(training)
    before = snapshot(model)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("optimizer setup failed")

    monkeypatch.setattr(torch.optim, "Adam", fail)
    with pytest.raises(RuntimeError, match="optimizer setup failed"):
        model.fit_supervised(X, Z, context=C, steps=1, context_only=context_only)
    assert calls == [1]
    assert_restored(model, before)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("context_only", [True, False])
def test_mode_setup_failure_restores_flags_and_mode(monkeypatch, training, context_only):
    model = make(training)
    before = snapshot(model)
    original = model.train
    calls = []

    def fail_once(mode=True):
        calls.append(mode)
        result = original(mode)
        if len(calls) == 1:
            raise RuntimeError("mode setup failed")
        return result

    monkeypatch.setattr(model, "train", fail_once)
    with pytest.raises(RuntimeError, match="mode setup failed"):
        model.fit_supervised(X, Z, context=C, steps=1, context_only=context_only)
    assert calls == [True, training]
    assert_restored(model, before)


@pytest.mark.parametrize("training", [False, True])
def test_empty_trainable_set_retains_all_original_flags(training):
    model = make(training)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = snapshot(model)
    with pytest.raises(ValueError, match="no trainable parameters"):
        model.fit_supervised(X, Z, context=C, steps=1, context_only=True)
    assert_restored(model, before)
