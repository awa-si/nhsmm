from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch


@dataclass(frozen=True)
class CausalPath:
    states: tuple[int, ...]
    ages: tuple[int, ...]
    log_prob: torch.Tensor


@dataclass(frozen=True)
class SegmentPath:
    states: tuple[int, ...]
    durations: tuple[int, ...]
    log_prob: torch.Tensor


def _require_log_prob_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if torch.isnan(value).any() or torch.isposinf(value).any():
        raise ValueError(f"{name} must not contain NaN or +inf")
    return value


def _normalize_last_dim(log_values: torch.Tensor, name: str) -> torch.Tensor:
    _require_log_prob_tensor(name, log_values)
    z = torch.logsumexp(log_values, dim=-1, keepdim=True)
    if not torch.isfinite(z).all():
        raise ValueError(f"{name} contains a row with zero probability mass")
    return log_values - z


def duration_hazards_from_pmf(log_duration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PMF-to-hazard transform from the Gate-0 equations."""
    log_p = _normalize_last_dim(log_duration, "log_duration")
    if log_p.ndim != 3:
        raise ValueError("log_duration must be [T,K,D]")
    T, K, D = log_p.shape
    log_end = torch.full_like(log_p, float("-inf"))
    log_continue = torch.full_like(log_p, float("-inf"))

    # Deliberately simple direct sums; do not mirror production logcumsumexp logic.
    for t in range(T):
        for k in range(K):
            probs = log_p[t, k].exp()
            for age_index in range(D):
                survival = probs[age_index:].sum()
                if survival <= 0:
                    continue
                p_end = probs[age_index] / survival
                if p_end > 0:
                    log_end[t, k, age_index] = p_end.log()
                if age_index + 1 < D:
                    p_continue = probs[age_index + 1 :].sum() / survival
                    if p_continue > 0:
                        log_continue[t, k, age_index] = p_continue.log()
    return log_end, log_continue


def enumerate_causal_paths(
    log_initial: torch.Tensor,
    log_duration: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> list[CausalPath]:
    """Enumerate all valid causal `(state, age)` paths exactly for one sequence.

    Shapes:
      log_initial: [K]
      log_duration: [T,K,D]
      log_transition: [T,K,D,K]
      log_emission: [T,K]
    """
    log_initial = _normalize_last_dim(log_initial, "log_initial")
    log_duration = _normalize_last_dim(log_duration, "log_duration")
    log_transition = _normalize_last_dim(log_transition, "log_transition")
    _require_log_prob_tensor("log_emission", log_emission)

    if log_initial.ndim != 1:
        raise ValueError("log_initial must be [K]")
    if log_duration.ndim != 3:
        raise ValueError("log_duration must be [T,K,D]")
    if log_transition.ndim != 4:
        raise ValueError("log_transition must be [T,K,D,K]")
    if log_emission.ndim != 2:
        raise ValueError("log_emission must be [T,K]")

    T, K = log_emission.shape
    if log_initial.shape != (K,):
        raise ValueError("initial/state shape mismatch")
    if log_duration.shape[:2] != (T, K):
        raise ValueError("duration shape mismatch")
    D = log_duration.shape[-1]
    if log_transition.shape != (T, K, D, K):
        raise ValueError("transition shape mismatch")
    if T == 0:
        return []

    log_end, log_continue = duration_hazards_from_pmf(log_duration)
    frontier: list[CausalPath] = []
    for state in range(K):
        lp = log_initial[state] + log_emission[0, state]
        if torch.isfinite(lp):
            frontier.append(CausalPath((state,), (1,), lp))

    for t in range(T - 1):
        next_frontier: list[CausalPath] = []
        for path in frontier:
            state = path.states[-1]
            age = path.ages[-1]
            age_index = age - 1

            if age < D and torch.isfinite(log_continue[t, state, age_index]):
                lp = path.log_prob + log_continue[t, state, age_index] + log_emission[t + 1, state]
                if torch.isfinite(lp):
                    next_frontier.append(
                        CausalPath(path.states + (state,), path.ages + (age + 1,), lp)
                    )

            if torch.isfinite(log_end[t, state, age_index]):
                for next_state in range(K):
                    trans = log_transition[t, state, age_index, next_state]
                    lp = path.log_prob + log_end[t, state, age_index] + trans
                    lp = lp + log_emission[t + 1, next_state]
                    if torch.isfinite(lp):
                        next_frontier.append(
                            CausalPath(path.states + (next_state,), path.ages + (1,), lp)
                        )
        frontier = next_frontier
    return frontier


def causal_log_likelihood(paths: Iterable[CausalPath]) -> torch.Tensor:
    paths = list(paths)
    if not paths:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return torch.logsumexp(torch.stack([p.log_prob for p in paths]), dim=0)


def causal_prefix_joint_posterior(
    log_initial: torch.Tensor,
    log_duration: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> torch.Tensor:
    """Exact `P(state_t, age_t | x_0:t)` for every prefix via path enumeration."""
    T, K = log_emission.shape
    D = log_duration.shape[-1]
    result = log_emission.new_full((T, K, D), float("-inf"))
    for end in range(T):
        paths = enumerate_causal_paths(
            log_initial,
            log_duration[: end + 1],
            log_transition[: end + 1],
            log_emission[: end + 1],
        )
        total = causal_log_likelihood(paths)
        for state in range(K):
            for age in range(1, D + 1):
                selected = [
                    p.log_prob for p in paths if p.states[-1] == state and p.ages[-1] == age
                ]
                if selected:
                    result[end, state, age - 1] = (
                        torch.logsumexp(torch.stack(selected), dim=0) - total
                    )
    return result


def _compositions(total: int, max_part: int) -> Iterable[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    for first in range(1, min(max_part, total) + 1):
        for rest in _compositions(total - first, max_part):
            yield (first,) + rest


def enumerate_segment_paths(
    log_initial: torch.Tensor,
    log_duration: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> list[SegmentPath]:
    """Enumerate all valid non-causal segmentations exactly for one sequence."""
    log_initial = _normalize_last_dim(log_initial, "log_initial")
    log_duration = _normalize_last_dim(log_duration, "log_duration")
    log_transition = _normalize_last_dim(log_transition, "log_transition")
    _require_log_prob_tensor("log_emission", log_emission)

    T, K = log_emission.shape
    D = log_duration.shape[-1]
    if log_initial.shape != (K,):
        raise ValueError("initial/state shape mismatch")
    if log_duration.shape != (T, K, D):
        raise ValueError("duration shape mismatch")
    if log_transition.shape != (T, K, D, K):
        raise ValueError("transition shape mismatch")

    paths: list[SegmentPath] = []
    for durations in _compositions(T, D):
        for states in product(range(K), repeat=len(durations)):
            end = -1
            score = log_initial[states[0]]
            valid = torch.isfinite(score)
            for segment_index, (state, duration) in enumerate(zip(states, durations)):
                start = end + 1
                end = start + duration - 1
                if segment_index > 0:
                    prev_state = states[segment_index - 1]
                    prev_duration = durations[segment_index - 1]
                    trans = log_transition[start - 1, prev_state, prev_duration - 1, state]
                    score = score + trans
                score = score + log_duration[end, state, duration - 1]
                score = score + log_emission[start : end + 1, state].sum()
                valid = valid & torch.isfinite(score)
            if bool(valid):
                paths.append(SegmentPath(tuple(states), tuple(durations), score))
    return paths


def segment_log_likelihood(paths: Iterable[SegmentPath]) -> torch.Tensor:
    paths = list(paths)
    if not paths:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return torch.logsumexp(torch.stack([p.log_prob for p in paths]), dim=0)


def segment_map_state_path(paths: Iterable[SegmentPath]) -> tuple[list[int], torch.Tensor]:
    paths = list(paths)
    if not paths:
        raise ValueError("paths must not be empty")
    best = max(paths, key=lambda p: float(p.log_prob))
    states: list[int] = []
    for state, duration in zip(best.states, best.durations):
        states.extend([state] * duration)
    return states, best.log_prob


def causal_map_state_age_path(
    paths: Iterable[CausalPath],
) -> tuple[list[int], list[int], torch.Tensor]:
    paths = list(paths)
    if not paths:
        raise ValueError("paths must not be empty")
    best = max(paths, key=lambda p: float(p.log_prob))
    return list(best.states), list(best.ages), best.log_prob


def causal_prefix_log_evidence(
    log_initial: torch.Tensor,
    log_duration: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> torch.Tensor:
    T = log_emission.shape[0]
    evidence = []
    for end in range(T):
        paths = enumerate_causal_paths(
            log_initial,
            log_duration[: end + 1],
            log_transition[: end + 1],
            log_emission[: end + 1],
        )
        evidence.append(causal_log_likelihood(paths))
    return torch.stack(evidence)


def causal_prefix_episode_end_probability(
    log_initial: torch.Tensor,
    log_duration: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> torch.Tensor:
    """Exact P(active episode ends before next observation | x_0:t)."""
    posterior = causal_prefix_joint_posterior(
        log_initial, log_duration, log_transition, log_emission
    )
    log_end, _ = duration_hazards_from_pmf(log_duration)
    T, K, D = posterior.shape
    result = posterior.new_zeros(T)
    for t in range(T):
        terms = []
        for state in range(K):
            for age_index in range(D):
                if torch.isfinite(posterior[t, state, age_index]) and torch.isfinite(
                    log_end[t, state, age_index]
                ):
                    terms.append(posterior[t, state, age_index] + log_end[t, state, age_index])
        if terms:
            result[t] = torch.logsumexp(torch.stack(terms), dim=0).exp()
    return result


def state_posterior_from_joint(log_joint: torch.Tensor) -> torch.Tensor:
    if log_joint.ndim != 3:
        raise ValueError("log_joint must be [T,K,D]")
    return torch.logsumexp(log_joint, dim=-1).exp()


def age_posterior_from_joint(log_joint: torch.Tensor) -> torch.Tensor:
    if log_joint.ndim != 3:
        raise ValueError("log_joint must be [T,K,D]")
    return torch.logsumexp(log_joint, dim=1).exp()


def causal_one_step_transition_forecast(
    log_joint_posterior: torch.Tensor,
    log_duration: torch.Tensor,
    log_transition: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Independent one-step latent forecast from an exact normalized posterior."""
    if log_joint_posterior.ndim != 2:
        raise ValueError("log_joint_posterior must be [K,D]")
    if log_duration.ndim != 2:
        raise ValueError("log_duration must be [K,D]")
    if log_transition.ndim != 3:
        raise ValueError("log_transition must be [K,D,K]")

    log_duration = _normalize_last_dim(log_duration, "log_duration")
    log_transition = _normalize_last_dim(log_transition, "log_transition")
    log_end, log_continue = duration_hazards_from_pmf(log_duration.unsqueeze(0))
    log_end = log_end[0]
    log_continue = log_continue[0]

    K, D = log_joint_posterior.shape
    boundary_joint = log_joint_posterior.new_zeros((K, K))
    continuation = log_joint_posterior.new_zeros(K)
    for src in range(K):
        for age_index in range(D):
            mass = log_joint_posterior[src, age_index].exp()
            if mass == 0:
                continue
            if torch.isfinite(log_continue[src, age_index]):
                continuation[src] += mass * log_continue[src, age_index].exp()
            if torch.isfinite(log_end[src, age_index]):
                end_mass = mass * log_end[src, age_index].exp()
                boundary_joint[src] += end_mass * log_transition[src, age_index].exp()

    next_episode = boundary_joint.sum(dim=0)
    episode_end = next_episode.sum()
    next_state_prior = continuation + next_episode
    diagonal_boundary = boundary_joint.diagonal().sum()
    return {
        "episode_end_probability": episode_end,
        "boundary_transition_joint": boundary_joint,
        "next_episode_state_joint": next_episode,
        "next_state_prior": next_state_prior,
        "state_change_probability": episode_end - diagonal_boundary,
    }


def duration_hazards_from_tail_mass(
    log_duration_mass: torch.Tensor,
    tail_end_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent bounded-tail mass-to-hazard transform.

    ``log_duration_mass[..., :-1]`` represents exact durations ``1..D-1``.
    The final mass is ``P(total_duration >= D)``. The final returned hazard
    bucket represents ``age >= D`` and uses ``tail_end_probability``.
    """
    log_mass = _normalize_last_dim(log_duration_mass, "log_duration_mass")
    if log_mass.ndim != 3:
        raise ValueError("log_duration_mass must be [T,K,D]")
    if not isinstance(tail_end_probability, torch.Tensor):
        raise TypeError("tail_end_probability must be a torch.Tensor")
    if tail_end_probability.shape != log_mass.shape[:2]:
        raise ValueError("tail_end_probability must be [T,K]")
    if not tail_end_probability.is_floating_point():
        raise TypeError("tail_end_probability must use a floating dtype")
    if not torch.isfinite(tail_end_probability).all():
        raise ValueError("tail_end_probability must be finite")
    if bool(((tail_end_probability <= 0) | (tail_end_probability > 1)).any()):
        raise ValueError("tail_end_probability must satisfy 0 < h <= 1")

    T, K, D = log_mass.shape
    log_end = torch.full_like(log_mass, float("-inf"))
    log_continue = torch.full_like(log_mass, float("-inf"))

    for t in range(T):
        for k in range(K):
            probs = log_mass[t, k].exp()
            for age_index in range(max(D - 1, 0)):
                survival = probs[age_index:].sum()
                if survival <= 0:
                    continue
                p_end = probs[age_index] / survival
                p_continue = probs[age_index + 1 :].sum() / survival
                if p_end > 0:
                    log_end[t, k, age_index] = p_end.log()
                if p_continue > 0:
                    log_continue[t, k, age_index] = p_continue.log()

            h_tail = tail_end_probability[t, k]
            log_end[t, k, D - 1] = h_tail.log()
            p_continue_tail = 1.0 - h_tail
            if p_continue_tail > 0:
                log_continue[t, k, D - 1] = p_continue_tail.log()

    return log_end, log_continue
