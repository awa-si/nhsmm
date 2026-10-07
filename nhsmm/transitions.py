from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from nhsmm.filtering import HSMMFilterState, _require_compatible, duration_log_hazard


@dataclass(frozen=True)
class HSMMTransitionForecast:
    """One-step causal transition quantities for the current HSMM posterior.

    ``boundary_transition_joint[b, i, j]`` is the unconditional probability
    that the current episode ends before the next observation and a new episode
    starts in latent state ``j`` after ending in latent state ``i``.

    ``next_state_prior[b, j]`` is ``P(z_{t+1}=j | F_t)`` before observing
    ``x_{t+1}``, combining episode continuation and boundary transitions.
    """

    episode_end_probability: torch.Tensor  # [B]
    boundary_transition_joint: torch.Tensor  # [B,K,K]
    next_episode_state_joint: torch.Tensor  # [B,K]
    next_state_prior: torch.Tensor  # [B,K]
    state_change_probability: torch.Tensor  # [B]

    def __post_init__(self) -> None:
        end = self.episode_end_probability
        joint = self.boundary_transition_joint
        next_episode = self.next_episode_state_joint
        prior = self.next_state_prior
        change = self.state_change_probability

        if end.ndim != 1:
            raise ValueError(f"episode_end_probability must be [B], got {end.shape}")
        if joint.ndim != 3 or joint.shape[0] != end.shape[0] or joint.shape[1] != joint.shape[2]:
            raise ValueError(f"boundary_transition_joint must be [B,K,K], got {joint.shape}")
        B, K, _ = joint.shape
        if next_episode.shape != (B, K):
            raise ValueError(f"next_episode_state_joint must be {(B, K)}, got {next_episode.shape}")
        if prior.shape != (B, K):
            raise ValueError(f"next_state_prior must be {(B, K)}, got {prior.shape}")
        if change.shape != (B,):
            raise ValueError(f"state_change_probability must be {(B,)}, got {change.shape}")

        reference = end
        for name, value in (
            ("boundary_transition_joint", joint),
            ("next_episode_state_joint", next_episode),
            ("next_state_prior", prior),
            ("state_change_probability", change),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match episode_end_probability dtype/device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")

        if not torch.isfinite(end).all():
            raise ValueError("episode_end_probability must be finite")
        eps = 32 * torch.finfo(end.dtype).eps
        for name, value in (
            ("episode_end_probability", end),
            ("boundary_transition_joint", joint),
            ("next_episode_state_joint", next_episode),
            ("next_state_prior", prior),
            ("state_change_probability", change),
        ):
            if (value < -eps).any() or (value > 1.0 + eps).any():
                raise ValueError(f"{name} must contain probabilities in [0,1]")

        atol = float(64 * torch.finfo(end.dtype).eps)
        boundary_end = joint.sum(dim=(1, 2))
        if not torch.allclose(boundary_end, end, atol=atol, rtol=0.0):
            raise ValueError("boundary_transition_joint must sum to episode_end_probability")
        if not torch.allclose(
            next_episode.sum(dim=-1),
            end,
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError("next_episode_state_joint must sum to episode_end_probability")
        if not torch.allclose(
            next_episode,
            joint.sum(dim=1),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(
                "next_episode_state_joint must equal destination-marginal boundary mass"
            )
        if not torch.allclose(
            prior.sum(dim=-1),
            torch.ones_like(end),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError("next_state_prior must be normalized")
        if (change > end + atol).any():
            raise ValueError("state_change_probability cannot exceed episode_end_probability")


def _normalize_transition(log_transition: torch.Tensor) -> torch.Tensor:
    if torch.isnan(log_transition).any() or torch.isposinf(log_transition).any():
        raise ValueError("transition_log_prob must not contain NaN or +inf")
    log_z = torch.logsumexp(log_transition, dim=-1, keepdim=True)
    if not torch.isfinite(log_z).all():
        raise ValueError("transition_log_prob has zero or non-finite probability mass")
    return log_transition - log_z


def _probability_tolerance(reference: torch.Tensor) -> float:
    return float(64 * torch.finfo(reference.dtype).eps)


def _require_normalized_posterior(log_posterior: torch.Tensor) -> None:
    log_z = torch.logsumexp(log_posterior.flatten(1), dim=1)
    atol = _probability_tolerance(log_posterior)
    if not torch.allclose(
        log_z,
        torch.zeros_like(log_z),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("state.log_posterior must be normalized over state and age")


def one_step_transition_forecast(
    state: HSMMFilterState,
    log_duration: torch.Tensor,
    transition_log_prob: torch.Tensor,
    tail_end_probability: Optional[torch.Tensor] = None,
) -> HSMMTransitionForecast:
    """Forecast the next latent-state transition using only ``F_t``.

    The duration and transition distributions are the boundary scores already
    conditioned on the current causal context. No future observation or future
    context is used.
    """

    if not isinstance(state, HSMMFilterState):
        raise TypeError("state must be an HSMMFilterState")

    posterior = state.log_posterior
    _require_normalized_posterior(posterior)
    B, K, D = posterior.shape
    _require_compatible(posterior, log_duration, "log_duration")
    _require_compatible(posterior, transition_log_prob, "transition_log_prob")
    if log_duration.shape != (B, K, D):
        raise ValueError(f"log_duration must be {(B, K, D)}, got {log_duration.shape}")
    if transition_log_prob.shape != (B, K, D, K):
        raise ValueError(
            f"transition_log_prob must be {(B, K, D, K)}, got {transition_log_prob.shape}"
        )

    log_transition = _normalize_transition(transition_log_prob)
    log_end, log_continue = duration_log_hazard(log_duration, tail_end_probability)

    # Joint boundary mass for source state -> destination state, marginalized over age.
    log_boundary_by_age = posterior + log_end
    log_boundary_joint = torch.logsumexp(
        log_boundary_by_age.unsqueeze(-1) + log_transition,
        dim=2,
    )  # [B,K_src,K_dst]
    boundary_joint = log_boundary_joint.exp()

    next_episode_joint = boundary_joint.sum(dim=1)
    episode_end = next_episode_joint.sum(dim=-1)

    # Continuations retain the current state; boundaries may move to any next state.
    log_continue_state = torch.logsumexp(posterior + log_continue, dim=-1)  # [B,K]
    continuation_state = log_continue_state.exp()
    next_state_prior = continuation_state + next_episode_joint

    diagonal_boundary = boundary_joint.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    state_change = episode_end - diagonal_boundary

    eps = _probability_tolerance(posterior)
    if not torch.allclose(
        next_state_prior.sum(dim=-1), torch.ones_like(episode_end), atol=eps, rtol=0.0
    ):
        raise ValueError("next_state_prior is not normalized")
    if not torch.allclose(boundary_joint.sum(dim=(1, 2)), episode_end, atol=eps, rtol=0.0):
        raise ValueError("boundary transition mass is inconsistent with episode-end probability")

    return HSMMTransitionForecast(
        episode_end_probability=episode_end.clamp(0.0, 1.0),
        boundary_transition_joint=boundary_joint.clamp(0.0, 1.0),
        next_episode_state_joint=next_episode_joint.clamp(0.0, 1.0),
        next_state_prior=next_state_prior.clamp(0.0, 1.0),
        state_change_probability=state_change.clamp(0.0, 1.0),
    )
