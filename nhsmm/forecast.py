"""Causal one-step predictive observation distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .runtime import HSMMFilterRuntime


@dataclass(frozen=True)
class Forecast:
    """Gaussian mixture for selected features of the next observation.

    ``component_mean`` is [B,K,M], ``component_covariance`` is [B,K,M,M],
    and ``next_state_prior`` is [B,K]. ``mean`` and ``covariance`` are the
    mixture moments [B,M] and [B,M,M]. Feature indices refer to model input.
    """

    feature_indices: tuple[int, ...]
    next_state_prior: torch.Tensor
    component_mean: torch.Tensor
    component_covariance: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor


@torch.inference_mode()
def forecast(
    runtime: "HSMMFilterRuntime",
    mask: torch.Tensor,
    *,
    next_context: torch.Tensor | None = None,
) -> Forecast:
    """Predict masked features at t+1, conditioned on observations through t.

    ``mask`` is a one-dimensional boolean target selector of length F: True
    means forecast this feature. It is unrelated to padding/context availability.
    The future observation is never supplied. External next-step context must
    already be known at the forecast origin; unknown future context is not
    estimated. This first version supports Gaussian emissions only.
    """
    from .runtime import HSMMFilterRuntime

    if not isinstance(runtime, HSMMFilterRuntime) or runtime.state is None:
        raise ValueError("runtime must have processed an observation")
    model = runtime.model
    if not model.config.causal:
        raise ValueError("forecast requires a causal model")
    if model.dist.emission.emission_type != "gaussian":
        raise ValueError("forecast requires Gaussian emissions")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if mask.ndim != 1 or mask.shape[0] != model.config.n_features or not mask.any():
        raise ValueError("mask must select at least one of F observation features")

    from .transitions import one_step_transition_forecast

    state = runtime.state
    prior = one_step_transition_forecast(
        state.filter_state,
        state.duration_log_prob,
        state.transition_log_prob,
        state.tail_end_probability,
    ).next_state_prior
    batch, states = prior.shape
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
        if next_context.shape != (batch, 1, model.config.context_dim):
            raise ValueError("next_context must be [B,1,context_dim]")
        if next_context.device != prior.device or next_context.dtype != prior.dtype:
            raise ValueError("next_context must match runtime device and dtype")
        if not torch.isfinite(next_context).all():
            raise ValueError("next_context must be finite")

    loc, cov = model.dist.emission(context=next_context)
    if loc.shape[0] == 1 and batch > 1:
        loc = loc.expand(batch, *loc.shape[1:])
        cov = cov.expand(batch, *cov.shape[1:])
    features = model.config.n_features
    if loc.shape != (batch, 1, states, features) or cov.shape != (
        batch,
        1,
        states,
        features,
        features,
    ):
        raise ValueError("unexpected Gaussian emission shape")
    if not torch.isfinite(loc).all() or not torch.isfinite(cov).all():
        raise ValueError("invalid Gaussian emission parameters")

    indices = mask.nonzero(as_tuple=True)[0].to(device=prior.device)
    selected_mean = loc[:, 0].index_select(-1, indices)
    selected_cov = cov[:, 0].index_select(-2, indices).index_select(-1, indices)
    if (selected_cov.diagonal(dim1=-2, dim2=-1) <= 0).any():
        raise ValueError("invalid Gaussian emission covariance")
    mean = (prior.unsqueeze(-1) * selected_mean).sum(dim=1)
    centered = selected_mean - mean.unsqueeze(1)
    covariance = (
        prior[..., None, None] * (selected_cov + centered.unsqueeze(-1) * centered.unsqueeze(-2))
    ).sum(dim=1)
    return Forecast(
        feature_indices=tuple(indices.tolist()),
        next_state_prior=prior,
        component_mean=selected_mean,
        component_covariance=selected_cov,
        mean=mean,
        covariance=covariance,
    )
