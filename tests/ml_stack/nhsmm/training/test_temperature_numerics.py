"""Independent scalar references and end-to-end temperature boundaries."""

import math
from decimal import Decimal, localcontext

import pytest
import torch
from nhsmm import NHSMM, ModelConfig
from nhsmm.models import base
from nhsmm.training import _scheduled_temperature


@pytest.mark.parametrize("steepness", [0.01, 10.0, 1000.0, 2880.0, 4000.0])
@pytest.mark.parametrize("iteration", [0, 1, 2, 3])
def test_schedule_matches_high_precision_reference(steepness, iteration):
    with localcontext() as ctx:
        ctx.prec = 100
        exponent = Decimal(str(steepness)) / 4 * (Decimal(iteration) - 2)
        expected = Decimal("0.3") + Decimal("0.7") / (1 + exponent.exp())
    actual = _scheduled_temperature(iteration, 4, 0.3, 1.0, steepness)
    assert actual == pytest.approx(float(expected), rel=3e-16, abs=0.0)


def test_ordinary_schedule_is_bit_exact_to_previous_arithmetic():
    for target in (0.5, 1.0, 2.6):
        minimum = 0.3 * target
        for it in range(40):
            previous = minimum + (target - minimum) / (1 + math.exp((10.0 / 40) * (it - 20)))
            assert _scheduled_temperature(it, 40, minimum, target, 10.0) == previous


def make():
    cfg = ModelConfig(
        2,
        1,
        max_duration=2,
        causal=True,
        n_init=1,
        max_iter=4,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        seed=71,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    return model


@pytest.mark.parametrize("mode", ["annealed", "constant"])
def test_extreme_steepness_completes_with_matching_objective_history(mode, monkeypatch):
    model = make()
    temperatures = []
    original = base.observation_log_likelihood

    def record(model, sequences, temperature=1.0):
        temperatures.append(temperature)
        return original(model, sequences, temperature)

    monkeypatch.setattr(base, "observation_log_likelihood", record)
    cfg = model.config.with_overrides(
        temperature_steepness=1e308,
        optimizer_temperature_mode=mode,
        optimizer_initialization="current",
    )
    model.optimize(torch.tensor([[-1.0], [0.0], [1.0], [0.5]]), cfg=cfg)
    expected = [1.0] * 4 if mode == "constant" else [1.0, 1.0, 0.65, 0.3]
    assert temperatures[0:8:2] == pytest.approx(expected)
    assert [h["scheduled_temperature"] for h in model._optimization_history[0]] == pytest.approx(
        expected
    )
    assert model.optimization_summary()["main_updates"] == 4
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_underflowing_zero_temperature_rejected_before_model_mutation():
    model = make()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = model.config.with_overrides(
        temperature_steepness=1e308, temperature_min_ratio=5e-324, optimizer_target_temperature=0.1
    )
    with pytest.raises(ValueError, match="strictly positive"):
        model.optimize(torch.zeros(4, 1), cfg=cfg)
    assert not model.training
    assert not hasattr(model, "_optimizer")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
