"""One-step return forecasts from a causal Gaussian NHSMM runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .runtime import HSMMFilterRuntime


@dataclass(frozen=True)
class ReturnForecast:
    """Forecast before observing the next return, in model observation units.

    All tensors have a leading batch dimension. ``next_state_prior`` is
    P(z[t+1] | observations through t). The two probabilities refer to the
    *next* observation, not a decoded state at the current timestep.
    """

    next_state_prior: torch.Tensor  # [B,K]
    expected_return: torch.Tensor  # [B]
    return_variance: torch.Tensor  # [B]
    probability_up: torch.Tensor  # [B], P(return > zero_return)
    probability_tail: torch.Tensor  # [B], P(abs(return-tail_center) > tail_threshold)


@torch.inference_mode()
def forecast_return(
    runtime: HSMMFilterRuntime,
    *,
    tail_threshold: float,
    zero_return: float = 0.0,
    tail_center: float = 0.0,
    next_context: torch.Tensor | None = None,
) -> ReturnForecast:
    """Forecast the next univariate Gaussian emission after ``runtime.step``.

    ``next_context`` is the emission context known *at the forecast origin*;
    for a lag-one return context it is the current observed return. Pass it
    explicitly for external-context models. Internal causal encoders use the
    runtime's saved next-emission context. Thresholds and outputs are in the
    model's observation units; callers must apply training-only scaling.
    """
    if not isinstance(runtime, HSMMFilterRuntime) or runtime.state is None:
        raise ValueError("runtime must have processed an observation")
    model = runtime.model
    if not model.config.causal:
        raise ValueError("forecast_return requires a causal model")
    if model.config.n_features != 1 or model.dist.emission.emission_type != "gaussian":
        raise ValueError("forecast_return requires one Gaussian observation feature")
    for name, value in (
        ("tail_threshold", tail_threshold),
        ("zero_return", zero_return),
        ("tail_center", tail_center),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if tail_threshold <= 0:
        raise ValueError("tail_threshold must be positive")

    prior = runtime.forecast_transition().next_state_prior
    B, K = prior.shape
    if next_context is None:
        next_context = runtime.state.last_encoder_context
        if next_context is None and (
            model.config.context_dim is not None or model.config.use_context_encoder
        ):
            raise ValueError("next_context is required for this model")
    else:
        if runtime.state.last_encoder_context is not None:
            raise ValueError("internal encoder context is owned by the runtime")
        if model.config.context_dim is None:
            raise ValueError("next_context requires an external-context model")
        if not isinstance(next_context, torch.Tensor):
            raise TypeError("next_context must be a tensor")
        if next_context.ndim == 2:
            next_context = next_context.unsqueeze(1)
        if next_context.shape != (B, 1, model.config.context_dim):
            raise ValueError("next_context must be [B,1,context_dim]")
        if next_context.device != prior.device or next_context.dtype != prior.dtype:
            raise ValueError("next_context must match runtime device and dtype")
        if not torch.isfinite(next_context).all():
            raise ValueError("next_context must be finite")

    loc, covariance = model.dist.emission(context=next_context)
    if loc.shape[0] == 1 and B > 1:
        loc = loc.expand(B, *loc.shape[1:])
        covariance = covariance.expand(B, *covariance.shape[1:])
    if loc.shape != (B, 1, K, 1) or covariance.shape != (B, 1, K, 1, 1):
        raise ValueError("unexpected Gaussian emission shape")
    mean = loc[:, 0, :, 0]
    variance = covariance[:, 0, :, 0, 0]
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("invalid Gaussian emission parameters")
    sd = variance.sqrt()
    cdf = torch.special.ndtr
    up = (prior * cdf((mean - zero_return) / sd)).sum(dim=-1)
    low = cdf((tail_center - tail_threshold - mean) / sd)
    high = cdf((mean - tail_center - tail_threshold) / sd)
    tail = (prior * (low + high)).sum(dim=-1)
    expected = (prior * mean).sum(dim=-1)
    predictive_variance = (prior * (variance + mean.square())).sum(dim=-1) - expected.square()
    return ReturnForecast(
        next_state_prior=prior,
        expected_return=expected,
        return_variance=predictive_variance,
        probability_up=up.clamp(0.0, 1.0),
        probability_tail=tail.clamp(0.0, 1.0),
    )
