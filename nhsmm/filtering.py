from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

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

    @classmethod
    def _from_normalized_unchecked(
        cls,
        log_posterior: torch.Tensor,
    ) -> "HSMMFilterState":
        """Build internal state after the caller has already normalized it."""

        state = object.__new__(cls)
        object.__setattr__(state, "log_posterior", log_posterior)
        return state

    @property
    def state_log_posterior(self) -> torch.Tensor:
        return torch.logsumexp(self.log_posterior, dim=-1)

    @property
    def state_posterior(self) -> torch.Tensor:
        return self.state_log_posterior.exp()

    @property
    def age_posterior(self) -> torch.Tensor:
        return torch.logsumexp(self.log_posterior, dim=-2).exp()


@dataclass(frozen=True)
class HSMMFilterTrace:
    """Causal filtering trace over a padded batch.

    ``log_posterior[b, t, k, a]`` stores the normalized posterior over
    current latent state and current episode age for every valid timestep.
    Timesteps beyond ``lengths[b]`` remain ``-inf``.
    """

    log_posterior: torch.Tensor  # [B, T, K, D]
    lengths: torch.Tensor  # [B]

    def __post_init__(self) -> None:
        value = self.log_posterior
        lengths = self.lengths
        if not isinstance(value, torch.Tensor):
            raise TypeError("log_posterior must be a torch.Tensor")
        if value.ndim != 4:
            raise ValueError(f"log_posterior must be [B,T,K,D], got {value.shape}")
        if not value.is_floating_point():
            raise TypeError("log_posterior must use a floating dtype")
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a torch.Tensor")
        if lengths.ndim != 1 or lengths.shape[0] != value.shape[0]:
            raise ValueError(f"lengths must be [B], got {lengths.shape}")
        if (lengths < 0).any() or (lengths > value.shape[1]).any():
            raise ValueError("lengths contain values outside the trace time dimension")

    @property
    def state_posterior(self) -> torch.Tensor:
        return torch.logsumexp(self.log_posterior, dim=-1).exp()


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


def _duration_log_hazard_from_log_p(log_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute hazards from an already normalized duration log-PMF."""

    _, _, D = log_p.shape
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


def duration_log_hazard(log_duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert duration scores into end/continue log hazards.

    ``log_duration`` may contain unnormalized log scores, but every state must
    retain at least one finite duration. ``-inf`` is permitted for impossible
    durations; NaN, +inf, and all-impossible state rows are rejected.
    """

    log_duration = _as_batched(log_duration, 3, "log_duration")
    log_p = _normalize_log_probs(log_duration, dim=-1, name="log_duration")
    return _duration_log_hazard_from_log_p(log_p)


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


def _filter_step_normalized(
    previous: HSMMFilterState,
    emission_log_prob: torch.Tensor,
    log_duration: torch.Tensor,
    transition_log_prob: torch.Tensor,
) -> HSMMFilterState:
    """Fast runtime step for model-produced normalized duration/transition scores.

    This is an internal kernel. Callers must provide the canonical model outputs
    from ``Duration.log_matrix`` and ``Transition.log_matrix``. The public
    ``filter_step`` remains the validated path for arbitrary external scores.
    """

    prev = previous.log_posterior
    B, K, D = prev.shape
    log_end, log_continue = _duration_log_hazard_from_log_p(log_duration)

    predicted = prev.new_full((B, K, D), float("-inf"))
    if D > 1:
        predicted[..., 1:] = prev[..., :-1] + log_continue[..., :-1]

    boundary = prev + log_end
    predicted[..., 0] = torch.logsumexp(
        boundary.unsqueeze(-1) + transition_log_prob,
        dim=(1, 2),
    )

    posterior = predicted + emission_log_prob.unsqueeze(-1)
    flat = posterior.flatten(1)
    log_z = torch.logsumexp(flat, dim=1, keepdim=True)
    posterior = (flat - log_z).reshape_as(posterior)
    return HSMMFilterState._from_normalized_unchecked(posterior)


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
    normalized_duration = _normalize_log_probs(
        log_duration,
        dim=-1,
        name="log_duration",
    )
    return _filter_step_normalized(
        previous,
        emission_log_prob,
        normalized_duration,
        transition_log_prob,
    )


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


def filter_model_sequence(
    model: Any,
    X: Union[torch.Tensor, list[torch.Tensor]],
    context: Optional[torch.Tensor] = None,
    temperature: Optional[float] = None,
) -> HSMMFilterTrace:
    """Run the causal `(state, age)` filter using an initialized NHSMM.

    The boundary decision between observations ``t-1`` and ``t`` uses the
    duration and transition distributions conditioned only on information
    available at ``t-1``. The observation at ``t`` enters only through its
    emission likelihood after that boundary propagation. This ordering avoids
    using ``x_t`` to decide whether the preceding episode had already ended.

    This function is intentionally stateless. It establishes the model-bound
    filtering semantics before a mutable live ``step()`` adapter is added.
    """

    config = getattr(model, "config", None)
    if not bool(getattr(config, "causal", False)):
        raise ValueError("causal model filtering requires ModelConfig.causal=True")
    if bool(getattr(model, "training", False)):
        raise RuntimeError("model must be in eval mode before causal filtering")
    if getattr(model, "dist", None) is None:
        raise RuntimeError("model distributions are not initialized")

    with torch.inference_mode():
        sequence = model._build_sequence_set(X, context=context)
        B, T, K = sequence.log_probs.shape
        D = int(model.dist.duration.max_duration)
        trace = sequence.log_probs.new_full((B, T, K, D), float("-inf"))

        if T == 0:
            return HSMMFilterTrace(trace, sequence.lengths.clone())

        initial_logits = model.dist.initial.log_matrix(
            context=sequence.canonical,
            temperature=temperature,
            T=T,
        )
        duration_logits = model.dist.duration.log_matrix(
            context=sequence.contexts,
            temperature=temperature,
            T=T,
            soft_dmax=model.duration_logits_bias,
        )
        transition_logits = model.dist.transition.log_matrix(
            context=sequence.contexts,
            temperature=temperature,
            T=T,
            soft_dmax=model.duration_logits_bias,
        )

        expected_initial = (B, 1, K)
        expected_duration = (B, T, K, D)
        expected_transition = (B, T, K, D, K)
        if initial_logits.shape != expected_initial:
            raise ValueError(
                f"initial logits must be {expected_initial}, got {initial_logits.shape}"
            )
        if duration_logits.shape != expected_duration:
            raise ValueError(
                f"duration logits must be {expected_duration}, got {duration_logits.shape}"
            )
        if transition_logits.shape != expected_transition:
            raise ValueError(
                f"transition logits must be {expected_transition}, got {transition_logits.shape}"
            )

        for b in range(B):
            length = int(sequence.lengths[b].item())
            if length == 0:
                continue

            state = initialize_filter(
                initial_logits[b, 0],
                sequence.log_probs[b, 0],
                D,
            )
            trace[b, 0] = state.log_posterior[0]

            for t in range(1, length):
                state = filter_step(
                    state,
                    sequence.log_probs[b, t],
                    duration_logits[b, t - 1],
                    transition_logits[b, t - 1],
                )
                trace[b, t] = state.log_posterior[0]

        return HSMMFilterTrace(trace, sequence.lengths.clone())
