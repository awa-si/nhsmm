"""Causal state-age and completed-segment forward recursions."""

from __future__ import annotations
from typing import Optional, Union
from nhsmm.context import context_options

import torch
from nhsmm.context import SequenceSet, ContextRouter
from nhsmm.config import NEG_INF
from nhsmm.filtering import (
    _causal_filter_step,
    _duration_log_hazard_from_log_p,
    _initialize_causal_frontier,
    _safe_logsumexp,
)
from nhsmm.models.segments import EmissionPrefix

# Bounded specialization for the measured two/three-state filtering-loss workload.
_SMALL_HMM_SCAN_MAX_STATES = 3


def _hmm_prefix_log_posteriors(initial, emission, transition):
    """Inclusive log-semiring matrix scan, retaining every causal prefix.

    doc:nhsmm.hmm-prefix-scan (docs/supervised.md#small-hmm-gradient-cost)
    Common matrix scale factors cancel in the final row normalization. Scaling
    each scan block avoids a large cumulative log-evidence offset. Stable tensor
    primitives retain exact autograd; no prototype scan API or detached state.
    Work O(T K^3 log T), graph depth O(log T); restricted to small K by caller.
    """
    K = initial.numel()
    first = (initial + emission[0]).expand(1, K, K)
    blocks = torch.cat((first, transition + emission[1:, None, :]), dim=0)

    def scale(values):
        mass = _safe_logsumexp(values, dim=(-2, -1), keepdim=True)
        if not bool(torch.isfinite(mass).all()):
            raise ValueError("causal evidence has zero or non-finite probability mass")
        return values - mass

    blocks = scale(blocks)
    offset = 1
    while offset < len(blocks):
        combined = _safe_logsumexp(
            blocks[:-offset].unsqueeze(-1) + blocks[offset:].unsqueeze(-3), dim=-2
        )
        blocks = torch.cat((blocks[:offset], scale(combined)), dim=0)
        offset *= 2
    return torch.log_softmax(blocks[:, 0, :], dim=-1)


def _forward_causal_hazard(
    self,
    X: SequenceSet,
    router: ContextRouter,
    temperature: Optional[float] = None,
    *,
    normalized: bool = False,
) -> torch.Tensor:
    """Unnormalized causal forward recursion over ``(state, episode_age)``.

    Internal normalized=True returns log prefix posteriors for differentiable
    supervised filtering scores, without adding/subtracting cumulative evidence.
    The default unnormalized forward contract is unchanged.

    Boundary ``t-1 -> t`` uses duration and transition quantities conditioned
    only on ``F_{t-1}``; ``x_t`` enters after propagation through its emission.
    """

    B, T, K = router.log_probs.shape[:3]
    D = int(self.dist.duration.max_duration)
    if T == 0:
        return router.log_probs.new_empty((B, 0, K, D))

    initial_logits = self.dist.initial.log_matrix(
        context=self._active_context(router.canonical),
        temperature=temperature,
        T=T,
        **context_options(getattr(router, "context_mask", None), canonical=True),
    )
    duration_logits = self.dist.duration.log_matrix(
        context=self._active_context(router.context),
        temperature=temperature,
        T=T,
        soft_dmax=self.duration_logits_bias,
        **context_options(getattr(router, "context_mask", None), canonical=False),
    )
    transition_logits = self.dist.transition.log_matrix(
        context=self._active_context(router.context),
        temperature=temperature,
        T=T,
        soft_dmax=self.duration_logits_bias,
        **context_options(getattr(router, "context_mask", None), canonical=False),
    )

    initial_logits = self._expand_batch(initial_logits, B, "initial logits")
    duration_logits = self._expand_batch(duration_logits, B, "duration logits")
    transition_logits = self._expand_batch(transition_logits, B, "transition logits")

    expected_initial = (B, 1, K)
    expected_duration = (B, T, K, D)
    if initial_logits.shape != expected_initial:
        raise ValueError(f"initial logits must be {expected_initial}, got {initial_logits.shape}")
    if duration_logits.shape != expected_duration:
        raise ValueError(
            f"duration logits must be {expected_duration}, got {duration_logits.shape}"
        )

    duration_dependent_transition = self.dist.transition.max_duration is not None
    expected_transition = (B, T, K, D, K) if duration_dependent_transition else (B, T, K, K)
    if transition_logits.shape != expected_transition:
        raise ValueError(
            f"transition logits must be {expected_transition}, got {transition_logits.shape}"
        )

    if (
        normalized
        and D == 1
        and not self.config.duration_tail
        and K <= _SMALL_HMM_SCAN_MAX_STATES
        and bool(torch.isfinite(duration_logits).all())
    ):
        matrices = (
            transition_logits[..., 0, :] if duration_dependent_transition else transition_logits
        )
        rows = []
        for b, length in enumerate(X.lengths.tolist()):
            padding = router.log_probs.new_full((T - length, K), float("-inf"))
            if length:
                posterior = _hmm_prefix_log_posteriors(
                    initial_logits[b, 0],
                    router.log_probs[b, :length],
                    matrices[b, : length - 1],
                )
                rows.append(torch.cat((posterior, padding), dim=0).unsqueeze(-1))
            else:
                rows.append(padding.unsqueeze(-1))
        return torch.stack(rows, dim=0)

    tail_probability_fn = getattr(self.dist.duration, "tail_probability", None)
    tail_end_probability = tail_probability_fn() if callable(tail_probability_fn) else None
    expanded_tail = None
    if tail_end_probability is not None:
        expanded_tail = (
            tail_end_probability.to(device=duration_logits.device, dtype=duration_logits.dtype)
            .view(1, 1, K)
            .expand(B, T, K)
            .reshape(B * T, K)
        )
    log_end, log_continue = _duration_log_hazard_from_log_p(
        duration_logits.reshape(B * T, K, D), expanded_tail
    )
    log_end = log_end.reshape(B, T, K, D)
    log_continue = log_continue.reshape(B, T, K, D)

    transition = (
        transition_logits if duration_dependent_transition else transition_logits.unsqueeze(3)
    )
    args = (
        initial_logits[:, 0],
        router.log_probs,
        transition,
        log_end,
        log_continue,
        X.lengths,
        tail_end_probability is not None,
    )
    if (
        normalized
        and (D > 1 or self.config.duration_tail)
        and torch.is_grad_enabled()
        and router.log_probs.device.type == "cpu"
        and router.log_probs.dtype in (torch.float32, torch.float64)
    ):
        from nhsmm.models.filter_grad import _CausalPrefixAdjoint

        return _CausalPrefixAdjoint.apply(*args)
    return _causal_prefix_serial(*args, normalized=normalized)


def _causal_prefix_serial(
    initial, emission, transition, log_end, log_continue, lengths, tail, *, normalized
):
    """Shared forward owner; transition uses [B,T,K,D-or-1,K]."""
    B, T, K = emission.shape
    D = log_end.shape[-1]
    # Unbind once so each indexed input has only a single time row.
    # Indexing the full history separately at every t would repeatedly
    # allocate full-history zeros in each IndexBackward operation.
    emission_steps = emission.unbind(dim=1)
    transition_steps = transition.unbind(dim=1)
    log_end_steps = log_end.unbind(dim=1)
    log_continue_steps = log_continue.unbind(dim=1)

    if bool((lengths == T).all()):
        # A fully populated batch needs no active-row indexing or padding copies.
        # Keep precisely the same step owner and evidence addition order.
        frontier, cumulative_log_z = _initialize_causal_frontier(initial, emission_steps[0], D)
        alpha_steps = [frontier if normalized else frontier + cumulative_log_z[:, None, None]]
        for t in range(1, T):
            frontier, step_log_z = _causal_filter_step(
                frontier,
                emission_steps[t],
                log_end_steps[t - 1],
                log_continue_steps[t - 1],
                transition_steps[t - 1],
                tail_enabled=tail,
            )
            if not normalized:
                cumulative_log_z = cumulative_log_z + step_log_z
            alpha_steps.append(
                frontier if normalized else frontier + cumulative_log_z[:, None, None]
            )
        return torch.stack(alpha_steps, dim=1)

    # Recurse on a normalized frontier and carry the cumulative evidence
    # separately. This preserves the public unnormalized-alpha contract while
    # preventing avoidable Float32 drift from repeatedly propagating a large
    # sequence-level log offset through logsumexp operations.
    frontier = emission.new_full((B, K, D), float("-inf"))
    cumulative_log_z = emission.new_zeros(B)
    alpha_steps: list[torch.Tensor] = []

    valid0 = lengths > 0
    normalized_initial, initial_log_z = _initialize_causal_frontier(
        initial[valid0], emission_steps[0][valid0], D
    )
    frontier[valid0] = normalized_initial
    if not normalized:
        cumulative_log_z[valid0] = initial_log_z
    initial_active = torch.nonzero(valid0, as_tuple=False).squeeze(-1)
    first_alpha = emission.new_full((B, K, D), float("-inf"))
    alpha_steps.append(
        first_alpha.index_copy(
            0,
            initial_active,
            (
                normalized_initial
                if normalized
                else normalized_initial + initial_log_z[:, None, None]
            ).to(dtype=first_alpha.dtype),
        )
    )

    for t in range(1, T):
        valid = t < lengths
        if not bool(valid.any()):
            padding = emission.new_full((B, K, D), float("-inf"))
            alpha_steps.extend([padding] * (T - len(alpha_steps)))
            break
        active = torch.nonzero(valid, as_tuple=False).squeeze(-1)

        transition = transition_steps[t - 1].index_select(0, active)
        next_frontier, step_log_z = _causal_filter_step(
            frontier[active],
            emission_steps[t].index_select(0, active),
            log_end_steps[t - 1].index_select(0, active),
            log_continue_steps[t - 1].index_select(0, active),
            transition,
            tail_enabled=tail,
        )
        frontier[active] = next_frontier
        if not normalized:
            cumulative_log_z[active] = cumulative_log_z[active] + step_log_z
        # Stack output rows once instead of copying the full history gradient
        # through an indexed table write at every recurrence step.
        alpha_t = emission.new_full((B, K, D), float("-inf"))
        alpha_steps.append(
            alpha_t.index_copy(
                0,
                active,
                (
                    next_frontier
                    if normalized
                    else next_frontier + cumulative_log_z[active, None, None]
                ).to(dtype=alpha_t.dtype),
            )
        )

    return torch.stack(alpha_steps, dim=1)


def forward(
    self,
    X: SequenceSet,
    context: Optional[Union[torch.Tensor, ContextRouter]] = None,
    temperature: Optional[float] = None,
    timestep: Optional[int] = None,
) -> torch.Tensor:
    self._require_distributions()

    router = (
        ContextRouter.from_tensor(X, context=context)
        if not isinstance(context, ContextRouter)
        else context
    )
    if self.config.causal:
        if timestep is not None:
            raise ValueError("timestep is not supported by the full causal forward recursion")
        return self._forward_causal_hazard(X, router, temperature=temperature)
    if router.log_probs.shape[1] == 0:
        B, _, K = router.log_probs.shape
        return router.log_probs.new_empty((B, 0, K, self.dist.duration.max_duration))

    Dmax = self.dist.duration.max_duration
    B, T, K = router.log_probs.shape[:3]
    device = router.log_probs.device

    kwargs = dict(
        soft_dmax=self.duration_logits_bias,
        temperature=temperature,
        timestep=timestep,
        T=T,
    )
    initial_logits = self.dist.initial.log_matrix(
        context=self._active_context(router.canonical),
        **kwargs,
        **context_options(getattr(router, "context_mask", None), canonical=True),
    )
    duration_logits = self.dist.duration.log_matrix(
        context=self._active_context(router.context),
        **kwargs,
        **context_options(getattr(router, "context_mask", None), canonical=False),
    )
    transition_logits = self.dist.transition.log_matrix(
        context=self._active_context(router.context),
        **kwargs,
        **context_options(getattr(router, "context_mask", None), canonical=False),
    )
    initial_logits = self._expand_batch(initial_logits, B, "initial logits")
    duration_logits = self._expand_batch(duration_logits, B, "duration logits")
    transition_logits = self._expand_batch(transition_logits, B, "transition logits")

    tail_probability_fn = getattr(self.dist.duration, "tail_probability", None)
    tail_end_probability = tail_probability_fn() if callable(tail_probability_fn) else None
    if tail_end_probability is not None:
        emission_prefix = EmissionPrefix.build(router.log_probs, dim=1)
        duration_dependent_transition = self.dist.transition.max_duration is not None
        alpha_steps: list[torch.Tensor] = []

        for t in range(T):
            valid = t < X.lengths
            if not bool(valid.any()):
                alpha_steps.append(router.log_probs.new_full((B, K, Dmax), float("-inf")))
                continue
            active = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            bucket_candidates: list[list[torch.Tensor]] = [[] for _ in range(Dmax)]

            for duration in range(1, t + 2):
                start = t - duration + 1
                bucket = min(duration, Dmax) - 1
                emit_sum = emission_prefix.segment((active, t + 1), (active, start))
                segment_score = (
                    self._segment_duration_log_score(
                        duration_logits[active, t], duration, tail_end_probability
                    )
                    + emit_sum
                )

                if start == 0:
                    predecessor = initial_logits[active, 0]
                else:
                    prev_t = start - 1
                    prev_alpha = alpha_steps[prev_t].index_select(0, active)
                    if duration_dependent_transition:
                        predecessor = _safe_logsumexp(
                            prev_alpha.unsqueeze(-1) + transition_logits[active, prev_t],
                            dim=(1, 2),
                        )
                    else:
                        prev_state = _safe_logsumexp(prev_alpha, dim=-1)
                        predecessor = _safe_logsumexp(
                            prev_state.unsqueeze(-1) + transition_logits[active, prev_t],
                            dim=1,
                        )

                bucket_candidates[bucket].append(predecessor + segment_score)

            empty_bucket = router.log_probs.new_full((active.numel(), K), float("-inf"))
            bucket_values = [
                (
                    _safe_logsumexp(torch.stack(candidates, dim=0), dim=0)
                    if candidates
                    else empty_bucket
                )
                for candidates in bucket_candidates
            ]
            alpha_active = torch.stack(bucket_values, dim=-1)
            alpha_t = router.log_probs.new_full((B, K, Dmax), float("-inf"))
            alpha_steps.append(alpha_t.index_copy(0, active, alpha_active))

        alpha = torch.stack(alpha_steps, dim=1)
        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)
        return alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), NEG_INF)

    emission_prefix = EmissionPrefix.build(router.log_probs, dim=1)

    d_range = torch.arange(1, Dmax + 1, device=device)
    t_range = torch.arange(T, device=device).unsqueeze(1)
    start_idx = (t_range - d_range + 1).clamp(min=0)
    end_idx = t_range + 1

    start_idx = start_idx.unsqueeze(0).unsqueeze(2).expand(B, T, K, Dmax)
    end_idx = end_idx.unsqueeze(0).unsqueeze(2).expand(B, T, K, Dmax)

    emit_sums = emission_prefix.gathered(end_idx, start_idx)

    alpha = router.log_probs.new_full((B, T, K, Dmax), float("-inf"))
    valid0 = X.lengths > 0
    alpha[valid0, 0, :, 0] = (
        initial_logits[valid0, 0] + duration_logits[valid0, 0, :, 0] + emit_sums[valid0, 0, :, 0]
    )

    for t in range(1, T):
        valid = t < X.lengths
        if not bool(valid.any()):
            break
        active = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        max_d = min(Dmax, t + 1)
        alpha_t = alpha.new_full((active.numel(), K, max_d), float("-inf"))

        for duration_index in range(max_d):
            duration = duration_index + 1
            start = t - duration + 1

            if start == 0:
                predecessor = initial_logits[active, 0]
            else:
                prev_t = start - 1
                prev_alpha = alpha[active, prev_t]
                if self.dist.transition.max_duration is None:
                    prev_state = _safe_logsumexp(prev_alpha, dim=-1)
                    trans = transition_logits[active, prev_t]
                    predecessor = _safe_logsumexp(
                        prev_state.unsqueeze(-1) + trans,
                        dim=1,
                    )
                else:
                    trans = transition_logits[active, prev_t]
                    predecessor = _safe_logsumexp(
                        prev_alpha.unsqueeze(-1) + trans,
                        dim=(1, 2),
                    )

            alpha_t[..., duration_index] = (
                predecessor
                + duration_logits[active, t, :, duration_index]
                + emit_sums[active, t, :, duration_index]
            )

        full_alpha = alpha.new_full((active.numel(), K, Dmax), float("-inf"))
        full_alpha[..., :max_d] = alpha_t
        alpha[active, t] = full_alpha

    length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)
    alpha = alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), NEG_INF)
    return alpha
