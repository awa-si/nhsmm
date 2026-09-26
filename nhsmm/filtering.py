from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HSMMFilterState:
    """Normalized causal posterior over current latent state and episode age.

    ``log_posterior[b, k, a]`` represents
    ``log P(z_t=k, age_t=a+1 | x_0:t)``.
    """

    log_posterior: torch.Tensor  # [B, K, D]

    def __post_init__(self) -> None:
        value = self.log_posterior
        if not isinstance(value, torch.Tensor):
            raise TypeError("log_posterior must be a torch.Tensor")
        if value.ndim != 3:
            raise ValueError(f"log_posterior must be [B,K,D], got {value.shape}")
        if not value.is_floating_point():
            raise TypeError("log_posterior must use a floating dtype")
        if any(size < 1 for size in value.shape):
            raise ValueError(f"log_posterior dimensions must be non-empty, got {value.shape}")
        if torch.isnan(value).any() or torch.isposinf(value).any():
            raise ValueError("log_posterior must not contain NaN or +inf")
        log_z = torch.logsumexp(value.flatten(1), dim=1)
        if not torch.isfinite(log_z).all():
            raise ValueError("log_posterior must contain finite probability mass per batch item")

    @property
    def state_log_posterior(self) -> torch.Tensor:
        return torch.logsumexp(self.log_posterior, dim=-1)

    @property
    def state_posterior(self) -> torch.Tensor:
        return self.state_log_posterior.exp()

    @property
    def age_posterior(self) -> torch.Tensor:
        return torch.logsumexp(self.log_posterior, dim=-2).exp()


def _as_batched(tensor: torch.Tensor, ndim: int, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if tensor.ndim == ndim - 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim - 1} or {ndim} dims, got {tensor.shape}")
    return tensor


def _require_compatible(reference: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} device {tensor.device} != expected {reference.device}")
    if tensor.dtype != reference.dtype:
        raise ValueError(f"{name} dtype {tensor.dtype} != expected {reference.dtype}")


def _normalize_log_probs(log_values: torch.Tensor, *, dim: int, name: str) -> torch.Tensor:
    if torch.isnan(log_values).any() or torch.isposinf(log_values).any():
        raise ValueError(f"{name} must not contain NaN or +inf")
    log_z = torch.logsumexp(log_values, dim=dim, keepdim=True)
    if not torch.isfinite(log_z).all():
        raise ValueError(f"{name} has zero or non-finite probability mass")
    return log_values - log_z


def _normalize_filter_posterior(log_values: torch.Tensor) -> torch.Tensor:
    flat = log_values.flatten(1)
    normalized = _normalize_log_probs(flat, dim=1, name="filter posterior")
    return normalized.reshape_as(log_values)


def duration_log_hazard(log_duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert duration scores into end/continue log hazards.

    ``log_duration`` may contain unnormalized log scores, but every state must
    retain at least one finite duration. ``-inf`` is permitted for impossible
    durations; NaN, +inf, and all-impossible state rows are rejected.
    """

    log_duration = _as_batched(log_duration, 3, "log_duration")
    log_p = _normalize_log_probs(log_duration, dim=-1, name="log_duration")
    _, _, D = log_p.shape

    # log_survival[..., a] = log P(duration >= a + 1)
    rev_cumsum = torch.logcumsumexp(log_p.flip(-1), dim=-1).flip(-1)
    log_end = torch.full_like(log_p, float("-inf"))
    valid_survival = torch.isfinite(rev_cumsum)
    end_values = torch.where(valid_survival, log_p - rev_cumsum, log_end)
    log_end = torch.where(torch.isfinite(log_p), end_values, log_end)

    log_continue = torch.full_like(log_end, float("-inf"))
    if D > 1:
        valid_continue = valid_survival[..., :-1] & torch.isfinite(rev_cumsum[..., 1:])
        continue_values = rev_cumsum[..., 1:] - rev_cumsum[..., :-1]
        log_continue[..., :-1] = torch.where(
            valid_continue,
            continue_values,
            log_continue[..., :-1],
        )

    return log_end, log_continue


def initialize_filter(
    log_initial: torch.Tensor,
    emission_log_prob: torch.Tensor,
    max_duration: int,
) -> HSMMFilterState:
    """Initialize a causal HSMM filter at the first observed timestep."""

    if not isinstance(max_duration, int) or isinstance(max_duration, bool) or max_duration < 1:
        raise ValueError("max_duration must be an integer >= 1")

    log_initial = _as_batched(log_initial, 2, "log_initial")
    emission_log_prob = _as_batched(emission_log_prob, 2, "emission_log_prob")
    _require_compatible(log_initial, emission_log_prob, "emission_log_prob")
    if log_initial.shape != emission_log_prob.shape:
        raise ValueError(
            f"initial/emission shape mismatch: {log_initial.shape} != {emission_log_prob.shape}"
        )
    if torch.isnan(log_initial).any() or torch.isposinf(log_initial).any():
        raise ValueError("log_initial must not contain NaN or +inf")
    if torch.isnan(emission_log_prob).any() or torch.isposinf(emission_log_prob).any():
        raise ValueError("emission_log_prob must not contain NaN or +inf")

    B, K = log_initial.shape
    posterior = log_initial.new_full((B, K, max_duration), float("-inf"))
    posterior[..., 0] = log_initial + emission_log_prob
    return HSMMFilterState(_normalize_filter_posterior(posterior))


def filter_step(
    previous: HSMMFilterState,
    emission_log_prob: torch.Tensor,
    log_duration: torch.Tensor,
    transition_log_prob: torch.Tensor,
) -> HSMMFilterState:
    """Advance a causal explicit-duration HSMM filter by one observation.

    Duration probabilities describe total episode duration ``1..D``.
    Transitions occur only when the previous episode ends. A self-transition
    starts a new episode of the same latent state with age reset to one.
    """

    if not isinstance(previous, HSMMFilterState):
        raise TypeError("previous must be an HSMMFilterState")
    prev = previous.log_posterior
    B, K, D = prev.shape

    emission_log_prob = _as_batched(emission_log_prob, 2, "emission_log_prob")
    log_duration = _as_batched(log_duration, 3, "log_duration")
    transition_log_prob = _as_batched(transition_log_prob, 4, "transition_log_prob")
    for name, tensor in (
        ("emission_log_prob", emission_log_prob),
        ("log_duration", log_duration),
        ("transition_log_prob", transition_log_prob),
    ):
        _require_compatible(prev, tensor, name)

    if emission_log_prob.shape != (B, K):
        raise ValueError(f"emission_log_prob must be {(B, K)}, got {emission_log_prob.shape}")
    if log_duration.shape != (B, K, D):
        raise ValueError(f"log_duration must be {(B, K, D)}, got {log_duration.shape}")
    if transition_log_prob.shape != (B, K, D, K):
        raise ValueError(
            f"transition_log_prob must be {(B, K, D, K)}, got {transition_log_prob.shape}"
        )
    if torch.isnan(emission_log_prob).any() or torch.isposinf(emission_log_prob).any():
        raise ValueError("emission_log_prob must not contain NaN or +inf")

    transition_log_prob = _normalize_log_probs(
        transition_log_prob,
        dim=-1,
        name="transition_log_prob",
    )
    log_end, log_continue = duration_log_hazard(log_duration)

    predicted = prev.new_full((B, K, D), float("-inf"))

    # Existing episode survives one more timestep: same state, age increments.
    if D > 1:
        predicted[..., 1:] = prev[..., :-1] + log_continue[..., :-1]

    # Existing episode ends, then transition into a new episode at age one.
    boundary = prev + log_end
    new_episode = torch.logsumexp(
        boundary.unsqueeze(-1) + transition_log_prob,
        dim=(1, 2),
    )  # [B, K_next]
    predicted[..., 0] = new_episode

    posterior = predicted + emission_log_prob.unsqueeze(-1)
    return HSMMFilterState(_normalize_filter_posterior(posterior))


def next_episode_end_probability(
    state: HSMMFilterState,
    log_duration: torch.Tensor,
) -> torch.Tensor:
    """Probability that the currently active episode ends before the next observation."""

    if not isinstance(state, HSMMFilterState):
        raise TypeError("state must be an HSMMFilterState")
    prev = state.log_posterior
    B, K, D = prev.shape
    log_duration = _as_batched(log_duration, 3, "log_duration")
    _require_compatible(prev, log_duration, "log_duration")
    if log_duration.shape != (B, K, D):
        raise ValueError(f"log_duration must be {(B, K, D)}, got {log_duration.shape}")

    log_end, _ = duration_log_hazard(log_duration)
    result = torch.logsumexp(prev + log_end, dim=(1, 2)).exp()
    if not torch.isfinite(result).all():
        raise ValueError("episode-end probability is non-finite")
    return result
