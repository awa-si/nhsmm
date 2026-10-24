"""Boundary-aware supervised cohorts and smoothed static boundary estimation."""

from __future__ import annotations

import copy

import torch

from nhsmm.config import (
    EPS,
    MAX_LOGITS,
    MIN_LOGITS,
    SUPERVISED_BOUNDARY_FLOAT32_ATOL,
    SUPERVISED_BOUNDARY_FLOAT64_ATOL,
)


def _cohort(states, starts_at_boundary, mode):
    if mode not in {"hmm", "hsmm"}:
        raise ValueError("boundary cohorts require mode=hmm or hsmm")
    if isinstance(states, torch.Tensor):
        if states.ndim not in {1, 2}:
            raise ValueError("state paths must have one or two dimensions")
        states = [states] if states.ndim == 1 else list(states.unbind(0))
    if not isinstance(states, (list, tuple)):
        raise ValueError("one state path per sequence required")
    flags = [starts_at_boundary] if isinstance(starts_at_boundary, bool) else starts_at_boundary
    if not isinstance(flags, (list, tuple)) or len(flags) != len(states):
        raise ValueError("explicit starts_at_boundary flag required for each sequence")
    paths, offsets, kept = [], [], []
    for index, (z, known) in enumerate(zip(states, flags, strict=True)):
        if not isinstance(known, bool):
            raise ValueError("starts_at_boundary flags must be boolean")
        if not isinstance(z, torch.Tensor) or z.dtype != torch.long or z.ndim != 1:
            raise ValueError("state paths must be one-dimensional torch.long tensors")
        if bool((z < 0).any()):
            raise ValueError("state labels must be nonnegative")
        offset = 0
        if len(z) and mode == "hsmm" and not known:
            changes = torch.nonzero(z[1:] != z[:-1], as_tuple=False).flatten()
            offset = int(changes[0]) + 1 if len(changes) else len(z)
        offsets.append(offset)
        if offset < len(z):
            paths.append(z[offset:])
            kept.append(index)
    if not paths:
        raise ValueError("no observations after establishing fresh episode boundaries")
    return paths, {
        "mode": mode,
        "input_lengths": [len(z) for z in states],
        "offsets": offsets,
        "kept_indices": kept,
        "rows_removed": sum(offsets),
        "rows": sum(len(z) for z in paths),
    }


def prepare_supervised_sequences(observations, states, *, starts_at_boundary, mode="hsmm"):
    """Training-only trimming; never apply label-based trimming to evaluation."""
    paths, report = _cohort(states, starts_at_boundary, mode)
    if isinstance(observations, torch.Tensor):
        if observations.ndim not in {2, 3}:
            raise ValueError("observations must have two or three dimensions")
        observations = [observations] if observations.ndim == 2 else list(observations.unbind(0))
    if not isinstance(observations, (list, tuple)) or len(observations) != len(
        report["input_lengths"]
    ):
        raise ValueError("one observation tensor per input state path required")
    for x, length in zip(observations, report["input_lengths"], strict=True):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or len(x) != length:
            raise ValueError("observations must be unpadded [T,F] tensors matching state paths")
        if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
            raise ValueError("observations must be finite floating tensors")
    xs = [observations[i][report["offsets"][i] :] for i in report["kept_indices"]]
    return xs, paths, report


def _encode(head, log_probability, allowed=None, gate=None):
    value = log_probability.clone()
    if gate is not None:
        value = value - gate
    allowed = torch.ones_like(value, dtype=torch.bool) if allowed is None else allowed
    value = value * head.log_temperature.detach().double().cpu().exp()
    low = value.masked_fill(~allowed, torch.inf).amin(-1, keepdim=True)
    high = value.masked_fill(~allowed, -torch.inf).amax(-1, keepdim=True)
    value = torch.where(allowed, value - (high + low) / 2, torch.zeros_like(value))
    if not bool(torch.isfinite(value).all()) or bool(
        ((value < MIN_LOGITS) | (value > MAX_LOGITS)).any()
    ):
        raise ValueError("regularized probability range cannot be represented by model logits")
    converted = value.to(head.logits)
    if not bool(torch.isfinite(converted).all()):
        raise ValueError("regularized logits overflow model dtype")
    return converted


def fit_supervised_boundaries(model, states, *, starts_at_boundary):
    """Refit static duration/transition laws using training labels only.

    Reuses fitted emissions/initial weights. Final episodes are right-censored.
    This is a smoothed count estimator, not an Adam step or likelihood guarantee.
    """
    from nhsmm.distributions.default import Duration, Initial, Transition, Emission

    cfg = model.config
    cfg.__post_init__()
    if (
        cfg.model_mode not in {"hmm", "hsmm"}
        or cfg.transition_type != "ergodic"
        or (cfg.model_mode == "hsmm" and cfg.allow_self_transitions)
    ):
        raise ValueError("boundary count fitting requires a static ergodic HMM or HSMM")
    if model.dist is None:
        raise RuntimeError("initialize or fit model distributions before boundary fitting")
    expected = (
        ("duration", Duration),
        ("initial", Initial),
        ("transition", Transition),
        ("emission", Emission),
    )
    if any(type(getattr(model.dist, name)) is not cls for name, cls in expected):
        raise ValueError("boundary count fitting requires canonical distributions")
    paths, cohort = _cohort(states, starts_at_boundary, cfg.model_mode)
    k, d = cfg.n_states, cfg.max_duration
    if any(bool((z >= k).any()) for z in paths):
        raise ValueError("state label outside model state range")
    ends = torch.zeros(k, d, dtype=torch.float64)
    continued = torch.zeros_like(ends)
    counts = torch.zeros(k, d, k, dtype=torch.float64)
    for path in paths:
        z = path.detach().cpu()
        n = len(z)
        time = torch.arange(n)
        starts = torch.ones(n, dtype=torch.bool)
        if cfg.model_mode != "hmm":
            starts[1:] = z[1:] != z[:-1]
        beginning = torch.where(starts, time, torch.zeros_like(time)).cummax(0).values
        ages = time - beginning
        if not cfg.duration_tail and bool((ages >= d).any()):
            raise ValueError("observed episode exceeds finite duration support")
        source, target, age = z[:-1], z[1:], ages[:-1].clamp_max(d - 1)
        ended = starts[1:]
        end_index = source[ended] * d + age[ended]
        continue_index = source[~ended] * d + age[~ended]
        transition_index = end_index * k + target[ended]
        ends += torch.bincount(end_index, minlength=k * d).reshape(k, d)
        continued += torch.bincount(continue_index, minlength=k * d).reshape(k, d)
        counts += torch.bincount(transition_index, minlength=k * d * k).reshape(k, d, k)
    alpha = cfg.supervised_duration_end_prior
    beta = cfg.supervised_duration_continue_prior
    hazard = (ends + alpha) / (ends + continued + alpha + beta)
    if not cfg.duration_tail:
        hazard[:, -1] = 1
    log_survival = torch.cat(
        (torch.zeros(k, 1, dtype=torch.float64), torch.log1p(-hazard[:, :-1]).cumsum(-1)), -1
    )
    log_q = log_survival.clone()
    if d > 1:
        log_q[:, :-1] += hazard[:, :-1].log()
    allowed = torch.ones(k, k, dtype=torch.bool)
    if cfg.model_mode == "hsmm":
        allowed.fill_diagonal_(False)
    mask = allowed[:, None, :].expand(k, d, k)
    prior = cfg.supervised_transition_prior
    pooled = counts.sum(1) + prior * allowed
    pooled /= pooled.sum(-1, keepdim=True)
    concentration = cfg.supervised_transition_pool_strength if cfg.model_mode == "hsmm" else 0.0
    probability = counts + prior * mask + concentration * pooled[:, None, :]
    probability /= probability.sum(-1, keepdim=True)
    safe_probability = torch.where(mask, probability, torch.ones_like(probability))
    candidate = copy.deepcopy(model)
    gate = candidate.duration_logits_bias.detach().double().cpu().sigmoid().clamp_min(EPS).log()
    q_logits = _encode(candidate.dist.duration, log_q, gate=gate)
    a_logits = _encode(candidate.dist.transition, safe_probability.log(), allowed=mask)
    with torch.no_grad():
        candidate.dist.duration.logits.copy_(q_logits)
        candidate.dist.transition.logits.copy_(a_logits)
        if cfg.duration_tail:
            candidate.dist.duration.tail_end_probability.copy_(
                hazard[:, -1].to(candidate.dist.duration.tail_end_probability)
            )
        actual_q = (
            candidate.dist.duration.log_matrix(soft_dmax=candidate.duration_logits_bias)[0, 0]
            .double()
            .cpu()
        )
        actual_a = candidate.dist.transition.log_matrix()[0, 0].double().cpu().exp()
        tolerance = (
            SUPERVISED_BOUNDARY_FLOAT32_ATOL
            if candidate.dist.duration.logits.dtype == torch.float32
            else SUPERVISED_BOUNDARY_FLOAT64_ATOL
        )
        if not torch.allclose(actual_q, log_q, atol=tolerance, rtol=0) or not torch.allclose(
            actual_a, probability, atol=tolerance, rtol=0
        ):
            raise ValueError("regularized targets fail actual model score verification")
        if cfg.duration_tail and not torch.allclose(
            candidate.dist.duration.tail_probability().double().cpu(),
            hazard[:, -1],
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError("regularized tail hazard cannot be represented")
        # All validation and score verification precedes mutation of the live model.
        model.dist.duration.logits.copy_(candidate.dist.duration.logits)
        model.dist.transition.logits.copy_(candidate.dist.transition.logits)
        if cfg.duration_tail:
            model.dist.duration.tail_end_probability.copy_(
                candidate.dist.duration.tail_end_probability
            )
    return {
        "estimator": "censored_beta_smoothed_hazards_and_pooled_dirichlet_transitions",
        "optimizer_updates": 0,
        "cohort": cohort,
        "end_counts": ends.tolist(),
        "continue_counts": continued.tolist(),
        "transition_counts": counts.tolist(),
        "duration_hazards": hazard.tolist(),
        "transition_probability": probability.tolist(),
        "priors": {
            "duration_end": alpha,
            "duration_continue": beta,
            "transition": prior,
            "transition_pool_strength": concentration,
        },
    }
