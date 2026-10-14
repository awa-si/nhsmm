import pytest
import torch

from nhsmm import ModelConfig, NHSMM


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_target_temperature(value):
    with pytest.raises(ValueError):
        ModelConfig(n_states=2, n_features=1, optimizer_target_temperature=value)


def test_temperature_controls_roundtrip_and_defaults():
    cfg = ModelConfig(n_states=2, n_features=1)
    assert cfg.optimizer_temperature_mode == "annealed"
    assert cfg.optimizer_target_temperature == 1.0
    fixed = cfg.with_overrides(
        optimizer_temperature_mode="constant", optimizer_target_temperature=0.7
    )
    assert ModelConfig.from_dict(fixed.to_dict()) == fixed
    with pytest.raises(ValueError):
        cfg.with_overrides(optimizer_temperature_mode="invalid")


@pytest.mark.parametrize("mode", ["constant", "annealed"])
def test_actual_optimizer_restart_target_and_history(mode):
    torch.set_num_threads(1)
    torch.manual_seed(401916)
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=4,
        causal=True,
        n_init=2,
        max_iter=3,
        dropout=0.0,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        seed=401916,
        optimizer_temperature_mode=mode,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    model.eval()
    x = [torch.randn(9, 1), torch.randn(6, 1)]
    model.optimize(x)
    assert not model.training and not model.dist.training
    assert len(model._optimization_history) == 2
    assert all(len(history) == 3 for history in model._optimization_history)
    assert model._restart_score_temperature == 1.0
    with torch.no_grad():
        public = float(model.log_likelihood(x, reduce=True))
    assert public == pytest.approx(max(model._restart_scores), abs=1e-5)
    for history in model._optimization_history:
        assert [row["completed_updates"] for row in history] == [0, 1, 2]
        assert all(row["target_temperature"] == 1.0 for row in history)
        if mode == "constant":
            assert all(row["scheduled_temperature"] == 1.0 for row in history)
            assert all(abs(row["scheduled_ll"] - row["target_ll"]) < 1e-5 for row in history)
        else:
            assert history[0]["scheduled_temperature"] > history[-1]["scheduled_temperature"]


def test_mutated_temperature_config_rejected_before_initialization():
    cfg = ModelConfig(n_states=2, n_features=1)
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    original = model.dist
    model.config.optimizer_target_temperature = 0
    with pytest.raises(ValueError):
        model.optimize(torch.zeros(3, 1))
    assert model.dist is original


def test_current_initialization_preserves_frozen_parameters():
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=4,
        causal=True,
        n_init=1,
        max_iter=2,
        dropout=0.0,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        optimizer_initialization="current",
        optimizer_temperature_mode="constant",
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.dist.duration.logits.requires_grad_(True)
    original = model.dist
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = model.dist.duration.logits.clone()
    model.optimize(torch.randn(2, 10, 1))
    assert model.dist is original
    assert all(
        torch.equal(frozen[name], p) for name, p in model.named_parameters() if not p.requires_grad
    )
    assert not torch.equal(before, model.dist.duration.logits)
    with pytest.raises(ValueError, match="n_init=1"):
        cfg.with_overrides(n_init=2)
