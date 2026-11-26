"""Behavioral contracts for centralized model/training tunables."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch
from nhsmm import NHSMM, ModelConfig, build_artifact, load_artifact, save_artifact
from nhsmm.config import default_value
from nhsmm.distributions import Duration

NEW_FIELDS = [
    "context_pool_scale",
    "context_pool_layer_norm",
    "context_attention_init_std",
    "context_hidden_min",
    "encoder_hidden_min",
    "encoder_hidden_max",
    "encoder_hidden_multiplier",
    "context_delta_scale",
    "context_max_delta",
    "context_layer_norm_eps",
    "context_allow_projection",
    "context_final_activation",
    "context_xavier_gain",
    "leaky_relu_slope",
    "init_jitter",
    "logits_init_std",
    "initial_bias_start",
    "initial_bias_end",
    "duration_bias_start",
    "duration_bias_end",
    "duration_init_decay",
    "transition_bias_diagonal",
    "transition_bias_off_diagonal",
    "emission_init_std",
    "emission_spread_extent",
    "emission_spread_noise",
    "emission_raw_log_var",
    "emission_variance_proxy",
    "studentt_raw_dof",
    "studentt_raw_scale",
    "studentt_kmeans_raw_scale",
    "studentt_raw_dof_floor",
    "kmeans_max_iter",
    "kmeans_atol",
    "kmeans_rtol",
    "cluster_variance_min",
    "cluster_variance_max",
    "supervised_steps",
    "supervised_lr",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "grad_clip_norm",
    "supervised_grad_clip_norm",
    "scheduler_factor",
    "scheduler_min_patience",
    "temperature_min_ratio",
    "temperature_steepness",
    "convergence_patience",
    "convergence_patience_scale",
    "convergence_auto_patience",
    "convergence_patience_max",
]


def make(**kw):
    model = NHSMM(ModelConfig(3, 1, causal=True, max_duration=4, seed=91, **kw), device="cpu")
    model.initialize_distributions()
    return model


def test_all_new_fields_json_roundtrip_strict_and_nonmutating():
    cfg = ModelConfig(3, 1)
    assert set(NEW_FIELDS) <= cfg.to_dict().keys()
    assert ModelConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert cfg.with_overrides(supervised_lr=0.013).supervised_lr == 0.013
    assert cfg.supervised_lr == default_value("supervised_lr")
    with pytest.raises(ValueError, match="unknown"):
        cfg.with_overrides(supervised_lrr=0.01)


@pytest.mark.parametrize("name", NEW_FIELDS)
def test_new_fields_reject_invalid_types_or_nonfinite(name):
    cfg = ModelConfig(3, 1)
    value = getattr(cfg, name)
    bad = [] if isinstance(value, str) else float("nan")
    with pytest.raises((ValueError, TypeError)):
        cfg.with_overrides(**{name: bad})


@pytest.mark.parametrize(
    "overrides",
    [
        {"adam_beta1": 1.0},
        {"adam_beta2": -0.1},
        {"scheduler_factor": 1.0},
        {"temperature_min_ratio": 0.0},
        {"context_layer_norm_eps": 0.0},
        {"supervised_steps": True},
        {"supervised_lr": 0.0},
        {"supervised_grad_clip_norm": 0.0},
        {"encoder_hidden_min": 65},
        {"cluster_variance_min": 6.0},
        {"convergence_patience_max": 1},
        {"context_final_activation": "bad"},
    ],
)
def test_invalid_ranges_fail_closed(overrides):
    with pytest.raises(ValueError):
        ModelConfig(3, 1, **overrides)


def test_duration_and_initial_tunables_control_probability_laws():
    m = make(init_jitter=0.0, logits_init_std=0.0, duration_init_decay=0.7)
    expected = torch.softmax(-0.7 * torch.arange(4.0), 0).expand(3, 4)
    torch.testing.assert_close(m.dist.duration.log_matrix().exp()[0, 0], expected)
    torch.testing.assert_close(m.dist.initial.log_matrix().exp().flatten(), torch.full((3,), 1 / 3))
    low = Duration(3, activation="tanh")
    assert low.max_duration == default_value("max_duration")


def test_biased_initialization_controls_actual_simplexes():
    m = make(
        init_jitter=0.0,
        initial_init_mode="biased",
        initial_bias_start=4.0,
        initial_bias_end=1.0,
        transition_init_mode="biased",
        transition_bias_diagonal=4.0,
        transition_bias_off_diagonal=1.0,
    )
    torch.testing.assert_close(
        m.dist.initial.log_matrix().exp().flatten(), torch.tensor([4.0, 2.5, 1.0]) / 7.5
    )
    torch.testing.assert_close(
        m.dist.transition.log_matrix().exp()[0, 0, :, 0, :],
        (torch.ones(3, 3) + 3 * torch.eye(3)) / 6,
    )


def test_context_settings_reach_every_head_and_survive_fresh():
    m = make(
        context_dim=2,
        context_hidden_min=7,
        context_allow_projection=False,
        context_delta_scale=1.2,
        context_max_delta=0.07,
        transition_context_max_delta=0.09,
        context_layer_norm_eps=0.003,
        context_final_activation="identity",
        leaky_relu_slope=0.2,
        activation="leaky_relu",
    )
    for d in (m.dist.initial, m.dist.duration, m.dist.transition, m.dist.emission):
        assert d._proj is None
        assert d.context_net[0].out_features == 7
        assert d.context_net[1].eps == 0.003
        assert d.context_net[2].negative_slope == 0.2
        assert float(d.delta_scale.detach()) == pytest.approx(1.2)
        assert d._apply_context(d.base, torch.ones(1, 1, 2) * 100).abs().max() <= d.max_delta + 1e-6
    assert m.dist.transition.max_delta == 0.09
    assert m.dist.emission.max_delta == 0.07
    fresh = m.dist.fresh()
    assert fresh.initial.context_net[1].eps == 0.003
    assert fresh.initial._proj is None


def test_emission_initialization_and_kmeans_bounds(monkeypatch):
    m = make(
        emission_spread_extent=2.0,
        emission_spread_noise=0.0,
        init_jitter=0.0,
        emission_variance_proxy=2.0,
        cluster_variance_min=0.2,
        cluster_variance_max=0.4,
        kmeans_max_iter=7,
        kmeans_atol=0.002,
        kmeans_rtol=0.003,
    )
    torch.testing.assert_close(
        m.dist.emission.mu[:, 0].sort().values, torch.tensor([-2.0, 0.0, 2.0])
    )
    torch.testing.assert_close(
        torch.nn.functional.softplus(m.dist.emission.log_var),
        torch.full((3, 1), torch.log(torch.tensor(3.0)).item()),
    )
    seen = {}

    def centers(x, k, n_iter, **kw):
        seen.update(n_iter=n_iter, **kw)
        return torch.tensor([[-10.0], [0.0], [10.0]])

    monkeypatch.setattr(m, "_kmeans_centers", centers)
    with torch.no_grad():
        m._initialize_emission_from_observations(
            torch.tensor([[-11.0], [-9.0], [-0.1], [0.1], [9.0], [11.0]]), cluster_variance=True
        )
    assert seen == {"n_iter": 7, "atol": 0.002, "rtol": 0.003}
    torch.testing.assert_close(
        torch.nn.functional.softplus(m.dist.emission.log_var), torch.tensor([[0.4], [0.2], [0.4]])
    )


def test_supervised_omitted_arguments_resolve_config_and_explicit_override_wins():
    cfg = dict(
        supervised_steps=3,
        allow_self_transitions=False,
        supervised_lr=0.017,
        adam_beta1=0.7,
        adam_beta2=0.95,
        adam_eps=0.002,
    )
    a, b = make(**cfg), make(**cfg)
    x = torch.tensor([[-1.0], [-0.8], [0.0], [0.1], [0.9], [1.0]])
    z = torch.tensor([0, 0, 1, 1, 2, 2])
    summary = a.fit_supervised(x, z)
    b.fit_supervised(x, z, steps=3, lr=0.017)
    assert summary["updates"] == 3
    for n, t in a.state_dict().items():
        torch.testing.assert_close(t, b.state_dict()[n], rtol=0, atol=0)
    assert b.fit_supervised(x, z, steps=1, lr=0.01)["updates"] == 1


def test_adam_configuration_reaches_supervised_optimizer(monkeypatch):
    m = make(
        supervised_steps=1,
        allow_self_transitions=False,
        adam_beta1=0.65,
        adam_beta2=0.98,
        adam_eps=0.004,
        supervised_grad_clip_norm=0.2,
    )
    original = torch.optim.Adam
    seen = {}

    def adam(params, **kw):
        seen.update(kw)
        return original(params, **kw)

    monkeypatch.setattr(torch.optim, "Adam", adam)
    m.fit_supervised(torch.tensor([[-1.0], [0.0], [1.0]]), torch.arange(3))
    assert seen == {"lr": m.config.supervised_lr, "betas": (0.65, 0.98), "eps": 0.004}


def test_encoder_automatic_width_and_normalization_are_configured():
    m = make(
        use_context_encoder=True,
        encoder_hidden_min=6,
        encoder_hidden_max=10,
        encoder_hidden_multiplier=3,
        context_layer_norm_eps=0.002,
        dropout=0.0,
    )
    assert m.encoder.encoder.hidden_dim == 6
    assert m.encoder.encoder.norm.eps == 0.002


def test_nondefault_artifact_roundtrip_preserves_predictions(tmp_path):
    m = make(
        use_context_encoder=True,
        context_dim=6,
        hidden_dim=6,
        dropout=0.0,
        context_layer_norm_eps=0.004,
        context_allow_projection=False,
        context_final_activation="identity",
        context_max_delta=0.13,
    )
    m.eval()
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    before = m.log_likelihood(x)
    path = save_artifact(m, tmp_path / "config.pt")
    loaded = load_artifact(path, require_causal=True)
    assert loaded.config.to_dict() == m.config.to_dict()
    torch.testing.assert_close(loaded.log_likelihood(x), before, rtol=0, atol=0)


def test_old_artifact_missing_new_fields_reconstructs_legacy_inference(tmp_path):
    m = make(context_dim=2)
    m.eval()
    payload = deepcopy(build_artifact(m))
    for name in NEW_FIELDS:
        payload["model_config"].pop(name)
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    loaded = load_artifact(path, require_causal=True)
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    ctx = torch.ones(1, 3, 2)
    torch.testing.assert_close(
        loaded.log_likelihood(x, context=ctx), m.log_likelihood(x, context=ctx), rtol=0, atol=0
    )


def test_optimizer_rejects_different_context_architecture():
    m = make(context_dim=2)
    with pytest.raises(ValueError, match="structurally incompatible"):
        m.optimize(torch.zeros(3, 1), cfg=m.config.with_overrides(context_max_delta=0.2))


def test_pooling_tunables_reach_encoder_and_artifact(tmp_path):
    m = make(
        use_context_encoder=True,
        context_dim=6,
        hidden_dim=6,
        dropout=0.0,
        context_pool_scale=0.4,
        context_pool_layer_norm=False,
        context_attention_init_std=0.02,
        context_layer_norm_eps=0.003,
        pool="attn",
    )
    assert m.encoder.context_scale == 0.4
    assert m.encoder.layer_norm is False
    assert m.encoder.attention_init_std == 0.02
    assert m.encoder.layer_norm_eps == 0.003
    m.eval()
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    path = save_artifact(m, tmp_path / "pool.pt")
    restored = load_artifact(path, require_causal=True)
    assert restored.encoder.context_scale == 0.4
    assert restored.encoder.layer_norm is False
    torch.testing.assert_close(m.log_likelihood(x), restored.log_likelihood(x), rtol=0, atol=0)


def test_latent_optimizer_receives_adam_scheduler_and_patience_tunables():
    m = make(
        n_init=1,
        max_iter=1,
        convergence_stop=False,
        verbose=False,
        adam_beta1=0.6,
        adam_beta2=0.97,
        adam_eps=0.003,
        grad_clip_norm=0.4,
        scheduler_factor=0.3,
        scheduler_min_patience=4,
        convergence_patience=5,
        convergence_patience_max=9,
        convergence_patience_scale=2.0,
        convergence_auto_patience=False,
        temperature_min_ratio=0.2,
        temperature_steepness=7.0,
    )
    m.fit(torch.tensor([[-1.0], [0.0], [1.0]]))
    assert m._optimizer.defaults["betas"] == (0.6, 0.97)
    assert m._optimizer.defaults["eps"] == 0.003
    c = m._convergence
    assert c.patience == 5 and c.patience_max == 9 and c.patience_scale == 2.0
    assert c.auto_patience is False
    assert c.scheduler.factor == 0.3 and c.scheduler.patience == 4
