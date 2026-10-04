from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Union

import torch

from nhsmm.filtering import HSMMFilterState, duration_log_hazard

HorizonInput = Union[int, torch.Tensor, Iterable[int]]


@dataclass(frozen=True)
class HSMMSurvivalForecast:
    """Causal forecast for the currently active latent episode.

    ``survival_probability[b, i]`` is the probability that the active episode
    at the current filtered timestep is still active after ``horizons[i]``
    future boundaries. ``end_within_probability`` is its complementary
    cumulative probability that the current episode ends within that horizon.

    The forecast conditions on the duration distribution available at the
    current information set. It does not assume or synthesize future context.
    """

    horizons: torch.Tensor  # [H], positive strictly increasing integer bars
    survival_probability: torch.Tensor  # [B,H]
    end_within_probability: torch.Tensor  # [B,H]

    def __post_init__(self) -> None:
        horizons = self.horizons
        survival = self.survival_probability
        end_within = self.end_within_probability

        if not isinstance(horizons, torch.Tensor):
            raise TypeError("horizons must be a torch.Tensor")
        if horizons.ndim != 1 or horizons.numel() < 1:
            raise ValueError("horizons must be a non-empty one-dimensional tensor")
        if horizons.dtype != torch.long:
            raise TypeError("horizons must use torch.long dtype")
        if (horizons < 1).any():
            raise ValueError("horizons must be >= 1")
        if horizons.numel() > 1 and not torch.all(horizons[1:] > horizons[:-1]):
            raise ValueError("horizons must be strictly increasing")

        for name, value in (
            ("survival_probability", survival),
            ("end_within_probability", end_within),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 2 or value.shape[1] != horizons.numel():
                raise ValueError(
                    f"{name} must be [B,H] with H={horizons.numel()}, got {value.shape}"
                )
            if not value.is_floating_point():
                raise TypeError(f"{name} must use a floating dtype")
            if value.device != horizons.device:
                raise ValueError(f"{name} must be on the same device as horizons")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if ((value < 0.0) | (value > 1.0)).any():
                raise ValueError(f"{name} must stay within [0,1]")

        if survival.shape != end_within.shape:
            raise ValueError("survival/end-within probability shapes must match")
        if survival.dtype != end_within.dtype:
            raise ValueError("survival/end-within probability dtypes must match")
        if not torch.allclose(
            survival + end_within,
            torch.ones_like(survival),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("survival and end-within probabilities must be complementary")


def _as_horizons(horizons: HorizonInput, *, device: torch.device) -> torch.Tensor:
    if isinstance(horizons, torch.Tensor):
        if horizons.ndim == 0:
            horizons = horizons.reshape(1)
        if horizons.ndim != 1 or horizons.numel() < 1:
            raise ValueError("horizons must be a non-empty scalar or one-dimensional tensor")
        if horizons.dtype == torch.bool or horizons.is_floating_point() or horizons.is_complex():
            raise TypeError("horizons must contain integers")
        result = horizons.to(device=device, dtype=torch.long)
    elif isinstance(horizons, Integral) and not isinstance(horizons, bool):
        result = torch.tensor([int(horizons)], device=device, dtype=torch.long)
    else:
        try:
            values = list(horizons)
        except TypeError as exc:
            raise TypeError("horizons must be an integer or iterable of integers") from exc
        if not values:
            raise ValueError("horizons must not be empty")
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise TypeError("horizons must contain integers")
        result = torch.tensor([int(value) for value in values], device=device, dtype=torch.long)

    if (result < 1).any():
        raise ValueError("horizons must be >= 1")
    if result.numel() > 1 and not torch.all(result[1:] > result[:-1]):
        raise ValueError("horizons must be strictly increasing")
    return result


def active_episode_survival_forecast(
    state: HSMMFilterState,
    log_duration: torch.Tensor,
    horizons: HorizonInput,
) -> HSMMSurvivalForecast:
    """Forecast survival/end-within-h for the currently active episode.

    For current episode age ``a`` and horizon ``h``, survival uses the product
    of the duration continuation hazards for ages ``a .. a+h-1``. This is
    equivalent to ``P(D >= a+h | D >= a)`` under the current duration PMF,
    using one-based episode ages. The result is then averaged over the current
    joint posterior ``P(z_t, age_t | F_t)``.

    Horizon one therefore matches the existing one-step episode-end hazard.
    Future context is intentionally not rolled forward; the current duration
    distribution is frozen for this conditional forecast.
    """

    if not isinstance(state, HSMMFilterState):
        raise TypeError("state must be an HSMMFilterState")
    posterior = state.log_posterior
    B, K, D = posterior.shape

    if not isinstance(log_duration, torch.Tensor):
        raise TypeError("log_duration must be a torch.Tensor")
    if not log_duration.is_floating_point():
        raise TypeError("log_duration must use a floating dtype")
    if log_duration.ndim == 2:
        log_duration = log_duration.unsqueeze(0)
    if log_duration.shape != (B, K, D):
        raise ValueError(f"log_duration must be {(B, K, D)}, got {log_duration.shape}")
    if log_duration.device != posterior.device:
        raise ValueError("log_duration device must match filter state")
    if log_duration.dtype != posterior.dtype:
        raise ValueError("log_duration dtype must match filter state")

    horizon_tensor = _as_horizons(horizons, device=posterior.device)
    _, log_continue = duration_log_hazard(log_duration)

    survival_by_horizon = []
    for horizon_tensor_value in horizon_tensor:
        horizon = int(horizon_tensor_value.item())
        component_log_survival = posterior.new_full((B, K, D), float("-inf"))

        # From age index ``a``, surviving h future boundaries requires valid
        # continuation at a, a+1, ..., a+h-1. Ages that would exceed D have
        # zero survival probability by construction.
        if horizon < D:
            for age_index in range(D - horizon):
                component_log_survival[..., age_index] = log_continue[
                    ..., age_index : age_index + horizon
                ].sum(dim=-1)

        log_survival = torch.logsumexp(
            posterior + component_log_survival,
            dim=(1, 2),
        )
        survival_by_horizon.append(log_survival.exp())

    survival = torch.stack(survival_by_horizon, dim=1).clamp(0.0, 1.0)
    end_within = (1.0 - survival).clamp(0.0, 1.0)

    if survival.shape[1] > 1:
        tolerance = 1e-6
        if (survival[:, 1:] > survival[:, :-1] + tolerance).any():
            raise ValueError("survival probability must be non-increasing with horizon")
        if (end_within[:, 1:] + tolerance < end_within[:, :-1]).any():
            raise ValueError("end-within probability must be non-decreasing with horizon")

    return HSMMSurvivalForecast(
        horizons=horizon_tensor,
        survival_probability=survival,
        end_within_probability=end_within,
    )
