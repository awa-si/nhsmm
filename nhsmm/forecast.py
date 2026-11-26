"""Causal one-step forecast of calibrated market-regime probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch

if TYPE_CHECKING:
    from .runtime import HSMMFilterRuntime
    from .survival import HSMMSurvivalForecast


REGIMES = ("unknown", "chaotic", "bull", "bear", "range")


@dataclass(frozen=True)
class Forecast:
    """Next-step regime probabilities [B,5] in REGIMES order.

    These are a projection of the next latent-state prior through a separately
    calibrated [K,5] state-to-regime probability matrix. The projection must be
    fitted using labels available only in the training window.

    duration contains conditional survival/end probabilities for the *current
    latent episode* at user-selected future bar horizons. It does not forecast
    how many bars a named market regime will persist.
    """

    probabilities: torch.Tensor
    duration: "HSMMSurvivalForecast"
    labels: tuple[str, ...] = REGIMES


def validate_regime_projection(model: object, projection: torch.Tensor) -> torch.Tensor:
    """Freeze a row-stochastic latent-state to market-regime projection."""
    if not isinstance(projection, torch.Tensor) or not projection.is_floating_point():
        raise TypeError("regime_projection must be a floating tensor")
    if projection.shape != (model.config.n_states, len(REGIMES)):
        raise ValueError("regime_projection must be [K,5] in REGIMES order")
    parameter = next(model.parameters())
    if projection.device != parameter.device or projection.dtype != parameter.dtype:
        raise ValueError("regime_projection must match model device and dtype")
    if not torch.isfinite(projection).all() or (projection < 0).any():
        raise ValueError("regime_projection must contain finite nonnegative probabilities")
    if not torch.allclose(projection.sum(-1), torch.ones_like(projection[:, 0]), rtol=0, atol=1e-6):
        raise ValueError("each regime_projection row must sum to one")
    return projection.detach().clone()


@torch.inference_mode()
def forecast(
    runtime: "HSMMFilterRuntime", *, duration_horizons: int | Iterable[int] = (1, 3, 6)
) -> Forecast:
    """Next-step regimes plus conditional survival of the current latent episode.

    Durations count observation steps (bars), not clock time. The episode
    survival is *not* the predicted persistence of the market-regime label:
    distinct latent states can map to the same named market regime.
    """
    from .runtime import HSMMFilterRuntime
    from .transitions import one_step_transition_forecast

    if not isinstance(runtime, HSMMFilterRuntime) or runtime.state is None:
        raise ValueError("runtime must have processed an observation")
    if not runtime.model.config.causal:
        raise ValueError("forecast requires a causal model")
    if runtime.regime_projection is None:
        raise ValueError("forecast requires a training-calibrated regime_projection")
    current_versions = tuple(t._version for t in runtime.model.state_dict(keep_vars=True).values())
    if runtime._projection_versions != current_versions:
        raise ValueError("regime_projection is stale after model weights changed; recalibrate")

    state = runtime.state
    prior = one_step_transition_forecast(
        state.filter_state,
        state.duration_log_prob,
        state.transition_log_prob,
        state.tail_end_probability,
    ).next_state_prior
    projection = runtime.regime_projection
    if projection.device != prior.device or projection.dtype != prior.dtype:
        raise ValueError("regime_projection no longer matches the runtime device and dtype")
    duration = runtime.forecast_survival(duration_horizons)
    return Forecast(probabilities=prior @ projection, duration=duration)
