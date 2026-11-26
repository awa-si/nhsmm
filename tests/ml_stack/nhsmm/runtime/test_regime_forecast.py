"""Forecast is restricted to causally projected next-bar Nautilus market regimes."""

import pytest
import torch
from nhsmm import NHSMM, REGIMES, Forecast, HSMMFilterRuntime, ModelConfig


def _model(mode="hmm"):
    config = ModelConfig.for_mode(mode, n_states=3, n_features=1, seed=81)
    model = NHSMM(config, device="cpu").to(dtype=torch.float64)
    model.initialize_distributions(jitter=0)
    model.eval()
    return model


def test_regime_forecast_fail_closed_without_calibration():
    model = _model()
    runtime = HSMMFilterRuntime(model)
    with pytest.raises(ValueError, match="processed"):
        runtime.forecast()
    runtime.step(torch.tensor([0.25], dtype=torch.float64))
    with pytest.raises(ValueError, match="regime_projection"):
        runtime.forecast()


@pytest.mark.parametrize("mode", ["hmm", "hsmm"])
def test_projected_forecast_has_fixed_five_regimes(mode):
    model = _model(mode)
    p = torch.tensor(
        [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.1, 0.6, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]],
        dtype=torch.float64,
    )
    runtime = HSMMFilterRuntime(model, regime_projection=p)
    p.zero_()
    runtime.step(torch.tensor([0.25], dtype=torch.float64))
    result = runtime.forecast()
    assert isinstance(result, Forecast)
    assert result.labels == REGIMES == ("unknown", "chaotic", "bull", "bear", "range")
    assert result.probabilities.shape == (1, 5)
    torch.testing.assert_close(result.probabilities.sum(-1), torch.ones(1, dtype=torch.float64))
    assert torch.isfinite(result.probabilities).all()
    assert (result.probabilities >= 0).all()
    assert result.duration.horizons.tolist() == [1, 3, 6]
    assert result.duration.survival_probability.shape == (1, 3)
    torch.testing.assert_close(
        result.duration.survival_probability + result.duration.end_within_probability,
        torch.ones(1, 3, dtype=torch.float64),
    )
    via_existing = runtime.forecast_survival((1, 3, 6))
    torch.testing.assert_close(
        result.duration.survival_probability, via_existing.survival_probability
    )


def test_bad_projection_rejected_before_market_processing():
    model = _model()
    for p in [
        torch.ones(3, 4, dtype=torch.float64) / 4,
        torch.ones(3, 5, dtype=torch.float64),
        torch.full((3, 5), -0.2, dtype=torch.float64),
    ]:
        with pytest.raises((ValueError, TypeError)):
            HSMMFilterRuntime(model, regime_projection=p)


def test_refit_invalidates_regime_projection():
    model = _model()
    p = torch.ones(3, 5, dtype=torch.float64) / 5
    runtime = HSMMFilterRuntime(model, regime_projection=p)
    runtime.step(torch.tensor([0.1], dtype=torch.float64))
    runtime.forecast()
    with torch.no_grad():
        next(model.parameters()).add_(0.0001)
    with pytest.raises(ValueError, match="stale.*recalibrate"):
        runtime.forecast()


def test_only_market_regime_forecast_exported():
    import nhsmm

    assert not hasattr(nhsmm, "forecast_return")
    assert not hasattr(nhsmm, "ReturnForecast")
    assert hasattr(nhsmm, "Forecast")


def test_duration_horizons_override_and_validation():
    model = _model("hsmm")
    runtime = HSMMFilterRuntime(
        model, regime_projection=torch.full((3, 5), 0.2, dtype=torch.float64)
    )
    runtime.step(torch.tensor([0.25], dtype=torch.float64))
    result = runtime.forecast(duration_horizons=(1, 2, 5, 9))
    assert result.duration.horizons.tolist() == [1, 2, 5, 9]
    assert (
        result.duration.survival_probability[:, 1:]
        <= result.duration.survival_probability[:, :-1] + 1e-10
    ).all()
    with pytest.raises(ValueError):
        runtime.forecast(duration_horizons=(2, 1))
