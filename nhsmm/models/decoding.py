"""MAP decoding over episode ages or completed segment durations."""

from __future__ import annotations
from typing import List, Optional, Union
from nhsmm.context import context_options

import torch
from nhsmm.context import SequenceSet, ContextRouter
from nhsmm.filtering import duration_log_hazard, _require_reachable_hazards
from nhsmm.models.segments import EmissionPrefix


def _viterbi_terminal_index(score: torch.Tensor) -> int:
    value, index = score.reshape(-1).max(dim=0)
    if not torch.isfinite(value):
        raise ValueError("no finite Viterbi path exists under the model support")
    return int(index.item())


def _viterbi_causal_hazard(
    self,
    X: SequenceSet,
    router: ContextRouter,
) -> List[torch.Tensor]:
    """MAP causal decoding over ``(state, episode_age)`` under dynamic hazards."""

    K = self.config.n_states
    D = int(self.dist.duration.max_duration)
    predicted: List[torch.Tensor] = []

    for b in range(router.log_probs.shape[0]):
        L = int(X.lengths[b].item())
        if L == 0:
            predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
            continue

        initial_logits = self.dist.initial.log_matrix(
            context=self._active_context(router.canonical[b : b + 1]),
            T=L,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1]
                ),
                canonical=True,
            ),
        )[0, 0]
        duration_logits = self.dist.duration.log_matrix(
            context=self._active_context(router.context[b : b + 1, :L]),
            T=L,
            soft_dmax=self.duration_logits_bias,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1, :L]
                ),
                canonical=False,
            ),
        )[0]
        transition_logits = self.dist.transition.log_matrix(
            context=self._active_context(router.context[b : b + 1, :L]),
            T=L,
            soft_dmax=self.duration_logits_bias,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1, :L]
                ),
                canonical=False,
            ),
        )[0]
        tail_end_probability = self.dist.duration.tail_probability()
        log_end, log_continue = duration_log_hazard(
            duration_logits, tail_end_probability=tail_end_probability
        )

        score = router.log_probs.new_full((L, K, D), float("-inf"))
        prev_state = torch.full((L, K, D), -1, dtype=torch.long, device=score.device)
        prev_age = torch.full((L, K, D), -1, dtype=torch.long, device=score.device)
        score[0, :, 0] = initial_logits + router.log_probs[b, 0]

        state_indices = torch.arange(K, device=score.device).view(K, 1)
        age_indices = torch.arange(max(D - 1, 0), device=score.device).view(1, -1)
        duration_dependent_transition = self.dist.transition.max_duration is not None

        for t in range(1, L):
            _require_reachable_hazards(score[t - 1], log_end[t - 1], log_continue[t - 1])
            if D > 1:
                continuation = score[t - 1, :, :-1] + log_continue[t - 1, :, :-1]
                score[t, :, 1:] = continuation + router.log_probs[b, t].unsqueeze(-1)
                prev_state[t, :, 1:] = state_indices.expand(K, D - 1)
                prev_age[t, :, 1:] = age_indices.expand(K, D - 1)

            # The final age bucket represents D+ with an enabled tail.
            # Compete with the incoming D-1 continuation for its MAP path.
            if tail_end_probability is not None:
                tail_score = (
                    score[t - 1, :, -1] + log_continue[t - 1, :, -1] + router.log_probs[b, t]
                )
                target_age = D - 1
                use_tail = tail_score > score[t, :, target_age]
                score[t, :, target_age] = torch.where(use_tail, tail_score, score[t, :, target_age])
                prev_state[t, :, target_age] = torch.where(
                    use_tail,
                    torch.arange(K, device=score.device),
                    prev_state[t, :, target_age],
                )
                prev_age[t, :, target_age] = torch.where(
                    use_tail,
                    torch.full((K,), target_age, device=score.device),
                    prev_age[t, :, target_age],
                )

            boundary = score[t - 1] + log_end[t - 1]
            if duration_dependent_transition:
                candidates = boundary.unsqueeze(-1) + transition_logits[t - 1]
                flat = candidates.reshape(K * D, K)
                boundary_score, flat_index = flat.max(dim=0)
                boundary_prev_state = flat_index // D
                boundary_prev_age = flat_index % D
            else:
                boundary_state, boundary_age = boundary.max(dim=-1)
                candidates = boundary_state.unsqueeze(-1) + transition_logits[t - 1]
                boundary_score, boundary_prev_state = candidates.max(dim=0)
                boundary_prev_age = boundary_age[boundary_prev_state]

            boundary_score = boundary_score + router.log_probs[b, t]
            if D == 1 and tail_end_probability is not None:
                use_boundary = boundary_score > score[t, :, 0]
                score[t, :, 0] = torch.maximum(score[t, :, 0], boundary_score)
                prev_state[t, :, 0] = torch.where(
                    use_boundary, boundary_prev_state, prev_state[t, :, 0]
                )
                prev_age[t, :, 0] = torch.where(use_boundary, boundary_prev_age, prev_age[t, :, 0])
            else:
                score[t, :, 0] = boundary_score
                prev_state[t, :, 0] = boundary_prev_state
                prev_age[t, :, 0] = boundary_prev_age

        flat_index = self._viterbi_terminal_index(score[L - 1])
        state = flat_index // D
        age = flat_index % D
        path = torch.empty(L, dtype=torch.long, device=score.device)

        for t in range(L - 1, -1, -1):
            path[t] = state
            if t == 0:
                break
            state_next = int(prev_state[t, state, age].item())
            age_next = int(prev_age[t, state, age].item())
            if state_next < 0 or age_next < 0:
                raise RuntimeError("causal Viterbi backpointer is incomplete")
            state, age = state_next, age_next

        predicted.append(path)

    return predicted


def _viterbi_noncausal_tail(
    self,
    X: SequenceSet,
    router: ContextRouter,
) -> List[torch.Tensor]:
    K = self.config.n_states
    Dmax = int(self.dist.duration.max_duration)
    device = router.log_probs.device
    predicted: List[torch.Tensor] = []
    duration_dependent_transition = self.dist.transition.max_duration is not None
    tail_end_probability = self.dist.duration.tail_probability()
    if tail_end_probability is None:
        raise RuntimeError("tail Viterbi requires an enabled duration tail")

    for b in range(router.log_probs.shape[0]):
        L = int(router.mask[b].sum())
        if L == 0:
            predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
            continue

        initial_logits = self.dist.initial.log_matrix(
            context=self._active_context(router.canonical[b : b + 1]),
            T=L,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1]
                ),
                canonical=True,
            ),
        )[0, 0]
        duration_logits = self.dist.duration.log_matrix(
            context=self._active_context(router.context[b : b + 1, :L]),
            T=L,
            soft_dmax=self.duration_logits_bias,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1, :L]
                ),
                canonical=False,
            ),
        )[0]
        transition_logits = self.dist.transition.log_matrix(
            context=self._active_context(router.context[b : b + 1, :L]),
            T=L,
            soft_dmax=self.duration_logits_bias,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1, :L]
                ),
                canonical=False,
            ),
        )[0]

        emit_log = router.log_probs[b, :L]
        emission_prefix = EmissionPrefix.build(emit_log, dim=0)

        if duration_dependent_transition:
            score = router.log_probs.new_full((L, K, Dmax), float("-inf"))
            prev_state = torch.full((L, K, Dmax), -1, dtype=torch.long, device=device)
            prev_bucket = torch.full((L, K, Dmax), -1, dtype=torch.long, device=device)
            segment_length = torch.zeros((L, K, Dmax), dtype=torch.long, device=device)

            for t in range(L):
                for duration in range(1, t + 2):
                    start_t = t - duration + 1
                    bucket = min(duration, Dmax) - 1
                    emit_sum = emission_prefix.segment(t + 1, start_t)
                    segment_score = (
                        self._segment_duration_log_score(
                            duration_logits[t], duration, tail_end_probability
                        )
                        + emit_sum
                    )

                    if start_t == 0:
                        candidate = initial_logits + segment_score
                        candidate_prev_state = torch.full((K,), -1, dtype=torch.long, device=device)
                        candidate_prev_bucket = torch.full(
                            (K,), -1, dtype=torch.long, device=device
                        )
                    else:
                        predecessor_t = start_t - 1
                        candidates = (
                            score[predecessor_t].unsqueeze(-1) + transition_logits[predecessor_t]
                        )
                        flat = candidates.reshape(K * Dmax, K)
                        predecessor_score, flat_index = flat.max(dim=0)
                        candidate = predecessor_score + segment_score
                        candidate_prev_state = flat_index // Dmax
                        candidate_prev_bucket = flat_index % Dmax

                    improve = candidate > score[t, :, bucket]
                    score[t, :, bucket] = torch.where(improve, candidate, score[t, :, bucket])
                    prev_state[t, :, bucket] = torch.where(
                        improve, candidate_prev_state, prev_state[t, :, bucket]
                    )
                    prev_bucket[t, :, bucket] = torch.where(
                        improve, candidate_prev_bucket, prev_bucket[t, :, bucket]
                    )
                    segment_length[t, :, bucket] = torch.where(
                        improve,
                        torch.full((K,), duration, dtype=torch.long, device=device),
                        segment_length[t, :, bucket],
                    )

            flat_index = self._viterbi_terminal_index(score[L - 1])
            state = flat_index // Dmax
            bucket = flat_index % Dmax
            segments = []
            t = L - 1
            while t >= 0:
                duration = int(segment_length[t, state, bucket].item())
                if duration < 1:
                    raise RuntimeError("non-causal tail Viterbi segment length is missing")
                start_t = t - duration + 1
                segments.append((start_t, t, state))
                if start_t == 0:
                    break
                state_next = int(prev_state[t, state, bucket].item())
                bucket_next = int(prev_bucket[t, state, bucket].item())
                if state_next < 0 or bucket_next < 0:
                    raise RuntimeError("non-causal tail Viterbi backpointer is incomplete")
                state, bucket = state_next, bucket_next
                t = start_t - 1

            segments.reverse()
            predicted.append(
                torch.cat(
                    [
                        router.log_probs.new_full((end - start + 1,), st, dtype=torch.long)
                        for start, end, st in segments
                    ]
                )[:L]
            )
            continue

        V = router.log_probs.new_full((L, K), float("-inf"))
        back_ptr = torch.full((L, K), -1, dtype=torch.long, device=device)
        best_dur = torch.zeros((L, K), dtype=torch.long, device=device)

        for t in range(L):
            for duration in range(1, t + 2):
                start_t = t - duration + 1
                emit_sum = emission_prefix.segment(t + 1, start_t)
                segment_score = (
                    self._segment_duration_log_score(
                        duration_logits[t], duration, tail_end_probability
                    )
                    + emit_sum
                )
                if start_t == 0:
                    candidate = initial_logits + segment_score
                    candidate_prev = torch.full((K,), -1, dtype=torch.long, device=device)
                else:
                    predecessor_t = start_t - 1
                    predecessor_scores = (
                        V[predecessor_t].unsqueeze(-1) + transition_logits[predecessor_t]
                    )
                    predecessor, predecessor_state = predecessor_scores.max(dim=0)
                    candidate = predecessor + segment_score
                    candidate_prev = predecessor_state

                improve = candidate > V[t]
                V[t] = torch.where(improve, candidate, V[t])
                back_ptr[t] = torch.where(improve, candidate_prev, back_ptr[t])
                best_dur[t] = torch.where(
                    improve,
                    torch.full((K,), duration, dtype=torch.long, device=device),
                    best_dur[t],
                )

        t = L - 1
        state = self._viterbi_terminal_index(V[t])
        segments = []
        while t >= 0:
            duration = int(best_dur[t, state].item())
            if duration < 1:
                raise RuntimeError("non-causal tail Viterbi segment length is missing")
            start_t = t - duration + 1
            segments.append((start_t, t, state))
            prev = int(back_ptr[t, state].item())
            t = start_t - 1
            if prev >= 0:
                state = prev

        segments.reverse()
        predicted.append(
            torch.cat(
                [
                    router.log_probs.new_full((end - start + 1,), st, dtype=torch.long)
                    for start, end, st in segments
                ]
            )[:L]
        )

    return predicted


def _viterbi(
    self, X: SequenceSet, context: Optional[Union[torch.Tensor, ContextRouter]] = None
) -> List[torch.Tensor]:

    K = self.config.n_states
    Dmax = self.dist.duration.max_duration

    router = (
        context
        if isinstance(context, ContextRouter)
        else ContextRouter.from_tensor(X, context=context)
    )
    if self.config.causal:
        return self._viterbi_causal_hazard(X, router)
    if self.dist.duration.tail_probability() is not None:
        return self._viterbi_noncausal_tail(X, router)

    B, T_max, _ = router.log_probs.shape
    device = router.log_probs.device

    predicted: List[torch.Tensor] = []
    durations_full = torch.arange(1, Dmax + 1, device=device)
    duration_dependent_transition = self.dist.transition.max_duration is not None

    for b in range(B):
        L = int(router.mask[b].sum())
        if L == 0:
            predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
            continue

        initial_logits = self.dist.initial.log_matrix(
            context=self._active_context(router.canonical[b : b + 1]),
            T=L,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1]
                ),
                canonical=True,
            ),
        )[0, 0]
        duration_logits = self.dist.duration.log_matrix(
            context=self._active_context(router.context[b : b + 1, :L]),
            T=L,
            soft_dmax=self.duration_logits_bias,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1, :L]
                ),
                canonical=False,
            ),
        )[0]
        transition_logits = self.dist.transition.log_matrix(
            context=self._active_context(router.context[b : b + 1, :L]),
            T=L,
            soft_dmax=self.duration_logits_bias,
            **context_options(
                (
                    None
                    if getattr(router, "context_mask", None) is None
                    else getattr(router, "context_mask", None)[b : b + 1, :L]
                ),
                canonical=False,
            ),
        )[0]

        emit_log = router.log_probs[b, :L]
        emission_prefix = EmissionPrefix.build(emit_log, dim=0)

        if duration_dependent_transition:
            # Keep the ending segment duration in the Viterbi state. A future
            # transition is conditioned on that duration, so collapsing it
            # before scoring the destination can discard the global MAP path.
            score = router.log_probs.new_full((L, K, Dmax), float("-inf"))
            prev_state = torch.full((L, K, Dmax), -1, dtype=torch.long, device=device)
            prev_duration = torch.full((L, K, Dmax), -1, dtype=torch.long, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1
                emit_sums = emission_prefix.segment(t + 1, starts.clamp_min(0)).T
                scores_dur = duration_logits[t, :, :max_d] + emit_sums

                for duration_index in range(max_d):
                    duration = duration_index + 1
                    start_t = t - duration + 1
                    segment_score = scores_dur[:, duration_index]

                    if start_t == 0:
                        score[t, :, duration_index] = initial_logits + segment_score
                        continue

                    predecessor_t = start_t - 1
                    candidates = (
                        score[predecessor_t].unsqueeze(-1) + transition_logits[predecessor_t]
                    )
                    flat = candidates.reshape(K * Dmax, K)
                    predecessor_score, flat_index = flat.max(dim=0)
                    score[t, :, duration_index] = predecessor_score + segment_score
                    prev_state[t, :, duration_index] = flat_index // Dmax
                    prev_duration[t, :, duration_index] = flat_index % Dmax

            flat_index = self._viterbi_terminal_index(score[L - 1])
            state = flat_index // Dmax
            duration_index = flat_index % Dmax
            segments = []
            t = L - 1
            while t >= 0:
                duration = duration_index + 1
                start_t = t - duration + 1
                segments.append((start_t, t, state))
                if start_t == 0:
                    break
                predecessor_t = start_t - 1
                state_next = int(prev_state[t, state, duration_index].item())
                duration_next = int(prev_duration[t, state, duration_index].item())
                if state_next < 0 or duration_next < 0:
                    raise RuntimeError("non-causal Viterbi backpointer is incomplete")
                state, duration_index = state_next, duration_next
                t = predecessor_t

            segments.reverse()
            path = torch.cat(
                [
                    router.log_probs.new_full((end - start_t + 1,), st, dtype=torch.long)
                    for start_t, end, st in segments
                ]
            )
            predicted.append(path[:L])
            continue

        V = router.log_probs.new_full((L, K), float("-inf"))
        back_ptr = torch.full((L, K), -1, dtype=torch.long, device=device)
        best_dur = torch.zeros((L, K), dtype=torch.long, device=device)

        for t in range(L):
            max_d = min(Dmax, t + 1)
            durations = durations_full[:max_d]
            starts = t - durations + 1
            emit_sums = emission_prefix.segment(t + 1, starts.clamp_min(0)).T
            scores_dur = duration_logits[t, :, :max_d] + emit_sums

            best_score_t = router.log_probs.new_full((K,), float("-inf"))
            best_prev_t = torch.full((K,), -1, dtype=torch.long, device=device)
            best_duration_t = torch.ones((K,), dtype=torch.long, device=device)

            for duration_index in range(max_d):
                duration = duration_index + 1
                start_t = t - duration + 1
                segment_score = scores_dur[:, duration_index]

                if start_t == 0:
                    candidate = initial_logits + segment_score
                    candidate_prev = torch.full((K,), -1, dtype=torch.long, device=device)
                else:
                    predecessor_t = start_t - 1
                    trans = transition_logits[predecessor_t]
                    predecessor_scores = V[predecessor_t].unsqueeze(-1) + trans
                    predecessor, predecessor_state = predecessor_scores.max(dim=0)
                    candidate = predecessor + segment_score
                    candidate_prev = predecessor_state

                improve = candidate > best_score_t
                best_score_t = torch.where(improve, candidate, best_score_t)
                best_prev_t = torch.where(improve, candidate_prev, best_prev_t)
                best_duration_t = torch.where(
                    improve,
                    torch.full_like(best_duration_t, duration),
                    best_duration_t,
                )

            V[t] = best_score_t
            back_ptr[t] = best_prev_t
            best_dur[t] = best_duration_t

        t = L - 1
        state = self._viterbi_terminal_index(V[t])
        segments = []
        while t >= 0:
            d = int(best_dur[t, state])
            start_t = max(0, t - d + 1)
            segments.append((start_t, t, state))
            prev = int(back_ptr[t, state])
            t = start_t - 1
            if prev >= 0:
                state = prev

        segments.reverse()
        path = torch.cat(
            [
                router.log_probs.new_full((end - start_t + 1,), st, dtype=torch.long)
                for start_t, end, st in segments
            ]
        )

        predicted.append(path[:L])
    return predicted
