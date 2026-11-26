"""Behavioral checks for alternate latent-optimizer configurations."""

import math

import pytest
import torch
from nhsmm import NHSMM, ModelConfig, load_artifact, save_artifact
from nhsmm.models import base


def make(**changes):
    cfg = ModelConfig(
        2,
        1,
        max_duration=2,
        causal=True,
        context_dim=2,
        n_init=1,
        max_iter=2,
        dropout=0.0,
        verbose=False,
        use_scheduler=False,
        convergence_stop=False,
        seed=71,
    ).with_overrides(**changes)
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    return model


@pytest.mark.parametrize(
    "override",
    [
        {"cnn_channels": 7},
        {"n_heads": 2},
        {"pool": "last"},
        {"encoder_hidden_min": 24},
        {"dropout": 0.1},
        {"min_covar": 0.002},
        {"init_jitter": 0.02},
        {"duration_init_decay": 0.2},
        {"kmeans_max_iter": 7},
        {"seed": 73},
        {"supervised_lr": 0.01},
        {"pad_value": -99.0},
        {"duration_tail_init_probability": 0.3},
    ],
)
def test_unsupported_override_fails_before_mutation(override):
    model = make()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="structurally incompatible"):
        model.optimize(torch.zeros(4, 1), cfg=model.config.with_overrides(**override))
    assert not model.training
    assert not hasattr(model, "_optimizer")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_override_temperature_matches_actual_objective_and_history(monkeypatch):
    model = make()
    temperatures = []
    original = base.observation_log_likelihood

    def record(model, sequences, temperature=1.0):
        temperatures.append(temperature)
        return original(model, sequences, temperature)

    monkeypatch.setattr(base, "observation_log_likelihood", record)
    cfg = model.config.with_overrides(
        max_iter=4,
        temperature_steepness=2.0,
        temperature_min_ratio=0.2,
        optimizer_initialization="current",
    )
    model.optimize(torch.tensor([[-1.0], [0.0], [1.0], [0.5]]), cfg=cfg)
    expected = [0.2 + 0.8 / (1 + math.exp(2.0 / 4 * (i - 2))) for i in range(4)]
    assert temperatures[0:8:2] == pytest.approx(expected)
    assert temperatures[1:8:2] == [1.0] * 4
    assert temperatures[-1] == 1.0
    assert [r["scheduled_temperature"] for r in model._optimization_history[0]] == pytest.approx(
        expected
    )
    assert model.config.temperature_steepness == 10.0


def test_override_adam_and_clipping_reach_main_and_refinement(monkeypatch, tmp_path):
    model = make()
    optimizers, clips = [], []
    adam, clip = torch.optim.Adam, torch.nn.utils.clip_grad_norm_

    def record_adam(*args, **kwargs):
        result = adam(*args, **kwargs)
        optimizers.append(result)
        return result

    def record_clip(parameters, max_norm, *args, **kwargs):
        clips.append(max_norm)
        return clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", record_adam)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    cfg = model.config.with_overrides(
        optimizer_initialization="current",
        transition_refine_steps=1,
        adam_beta1=0.6,
        adam_beta2=0.97,
        adam_eps=0.003,
        grad_clip_norm=0.4,
    )
    x = torch.tensor([[-1.0], [0.0], [1.0], [0.5]])
    context = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]]])
    original_config = model.config.to_dict()
    model.optimize(x, context=context, cfg=cfg)
    assert len(optimizers) == 2
    for optimizer in optimizers:
        assert optimizer.defaults["betas"] == (0.6, 0.97)
        assert optimizer.defaults["eps"] == 0.003
    assert clips == [0.4, 0.4, 0.4]
    assert model.config.to_dict() == original_config
    model.eval()
    restored = load_artifact(save_artifact(model, tmp_path / "model.pt"), require_causal=True)
    torch.testing.assert_close(
        restored.log_likelihood(x, context=context),
        model.log_likelihood(x, context=context),
        rtol=0,
        atol=0,
    )
