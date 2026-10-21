"""Fully observed causal state paths, with a right-censored final episode.

doc:nhsmm.supervised-path (docs/supervised.md)
"""

from __future__ import annotations

import math

from nhsmm.context import context_options

import torch

from nhsmm.context import BASE_CONTEXT
from nhsmm.filtering import _duration_log_hazard_from_log_p, _safe_logsumexp


def _paths(labels, lengths, device, n_states):
    if isinstance(labels, torch.Tensor):
        labels = [labels] if labels.ndim == 1 else list(labels.unbind(0))
    if not isinstance(labels, (list, tuple)) or len(labels) != len(lengths):
        raise ValueError("one state-label tensor per sequence required")
    result = []
    for path, length in zip(labels, lengths.tolist(), strict=True):
        if not isinstance(path, torch.Tensor) or path.dtype != torch.long:
            raise ValueError("state labels must be torch.long tensors")
        if path.ndim != 1 or len(path) != length:
            raise ValueError("state labels must match the unpadded sequence length")
        if bool(((path < 0) | (path >= n_states)).any()):
            raise ValueError("state label outside model state range")
        result.append(path.to(device=device))
    return result


def supervised_log_likelihood(model, observations, labels, *, context=None, reduce=False):
    """Log p(x,z | external context), not log p(z|x) or the marginal p(x).

    Every sequence starts a new episode. No end event is observed after its last
    row. For D=1 without a tail, every edge is a new episode (ordinary HMM).
    Otherwise state labels determine episodes only when self resets are excluded.
    A zero-probability path returns -inf; fitting rejects it.
    """
    if not model.config.causal:
        raise ValueError("supervised paths require causal=True")
    hmm = model.config.max_duration == 1 and not model.config.duration_tail
    if not hmm and model.config.allow_self_transitions:
        raise ValueError("HSMM state supervision requires allow_self_transitions=False")
    data = model._build_sequence_set(observations, context=context)
    B, T, K = data.log_probs.shape
    paths = _paths(labels, data.lengths, data.sequences.device, K)
    if T == 0:
        scores = model.duration_logits_bias.sum() * 0 + data.sequences.new_zeros(B)
        return scores.sum() if reduce else scores
    active = model._active_context
    initial = model._expand_batch(
        model.dist.initial.log_matrix(
            context=active(data.canonical), **context_options(data.context_mask, canonical=True)
        ),
        B,
        "initial",
    )
    q = model._expand_batch(
        model.dist.duration.log_matrix(
            context=active(data.contexts),
            T=T,
            soft_dmax=model.duration_logits_bias,
            **context_options(data.context_mask, canonical=False),
        ),
        B,
        "duration",
    )
    transition = model._expand_batch(
        model.dist.transition.log_matrix(
            context=active(data.contexts),
            T=T,
            **context_options(data.context_mask, canonical=False),
        ),
        B,
        "transition",
    )
    D = model.config.max_duration
    tail = model.dist.duration.tail_probability()
    if tail is not None:
        tail = tail.view(1, 1, K).expand(B, T, K).reshape(B * T, K)
    end, continuation = _duration_log_hazard_from_log_p(q.reshape(B * T, K, D), tail)
    end, continuation = end.reshape(B, T, K, D), continuation.reshape(B, T, K, D)
    scores = []
    for b, z in enumerate(paths):
        n = len(z)
        if n == 0:
            scores.append(model.duration_logits_bias.sum() * 0)
            continue
        time = torch.arange(n, device=z.device)
        score = initial[b, 0, z[0]] + data.log_probs[b, time, z].sum()
        if n > 1:
            starts = torch.ones(n, dtype=torch.bool, device=z.device)
            if not hmm:
                starts[1:] = z[1:] != z[:-1]
            episode_start = torch.where(starts, time, torch.zeros_like(time)).cummax(0).values
            ages = time - episode_start
            if not model.config.duration_tail and bool((ages >= D).any()):
                scores.append(score + score.new_tensor(float("-inf")))
                continue
            age = ages[:-1].clamp_max(D - 1)
            previous, current, t = z[:-1], z[1:], time[:-1]
            ended = starts[1:]
            score = score + end[b, t[ended], previous[ended], age[ended]].sum()
            score = score + continuation[b, t[~ended], previous[~ended], age[~ended]].sum()
            if transition.ndim == 5:
                score = (
                    score
                    + transition[b, t[ended], previous[ended], age[ended], current[ended]].sum()
                )
            else:
                score = score + transition[b, t[ended], previous[ended], current[ended]].sum()
        scores.append(score)
    values = torch.stack(scores) if scores else data.sequences.new_empty(0)
    return values.sum() if reduce else values


def supervised_filter_log_score(model, observations, labels, *, context=None, reduce=False):
    """Sum log P(z_t | x_0:t, causal context), with predicted-state propagation.

    doc:nhsmm.supervised-filter (docs/supervised.md#causal-filtering-objective)
    This proper per-row classification score is not a joint path likelihood.
    Ages and unobserved episode resets are marginalized, including D+ tails.
    Every supplied sequence resets. Labels only gather scores; they never enter
    the recurrence, so self resets need not be identified by state labels.
    """
    from nhsmm.context import ContextRouter
    from nhsmm.models.forward import _forward_causal_hazard

    if not model.config.causal:
        raise ValueError("filter supervision requires a causal model")
    data = model._build_sequence_set(observations, context=context)
    B, T, K = data.log_probs.shape
    paths = _paths(labels, data.lengths, data.sequences.device, K)
    if T == 0 or B == 0:
        values = model.duration_logits_bias.sum() * 0 + data.sequences.new_zeros(B)
    else:
        boundary_inputs = []
        log_p = _safe_logsumexp(
            _forward_causal_hazard(
                model,
                data,
                ContextRouter.from_tensor(data),
                normalized=True,
                boundary_inputs=boundary_inputs,
            ),
            dim=-1,
        )
        values = torch.stack(
            [log_p[b, : len(z)].gather(1, z[:, None]).sum() for b, z in enumerate(paths)]
        )
        from nhsmm.models.boundary import tail_boundary_filter_score

        values = tail_boundary_filter_score(model, paths, values, boundary_inputs)
    return values.sum() if reduce else values


def fit_supervised(
    model,
    observations,
    labels,
    *,
    context=None,
    steps=None,
    lr=None,
    context_only=False,
    objective="observed_joint_causal_prefix",
):
    """Bounded full-batch Adam on an explicit supervised score; no latent fit.

    Uses current parameters, no hidden restarts, no validation or label matching.
    Returns a separate summary and leaves the latent optimization API untouched.
    """
    model.config.__post_init__()
    steps = model.config.supervised_steps if steps is None else steps
    lr = model.config.supervised_lr if lr is None else lr
    scorers = {
        "observed_joint_causal_prefix": supervised_log_likelihood,
        "causal_filter_log_score": supervised_filter_log_score,
    }
    if not isinstance(objective, str) or objective not in scorers:
        raise ValueError("unsupported supervised objective")
    score_fn = scorers[objective]
    if objective == "causal_filter_log_score" and not model.config.causal:
        raise ValueError("filter supervision requires a causal model")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if model.dist is None:
        model.initialize_distributions()
    # Validation before constructing an optimizer or changing a parameter.
    data = model._build_sequence_set(observations, context=context)
    count = int(data.lengths.sum())
    if count == 0:
        raise ValueError("supervised fitting requires observations")
    initial = score_fn(model, observations, labels, context=context, reduce=True)
    if not bool(torch.isfinite(initial)):
        raise ValueError("supervised path has zero or non-finite probability")
    if not isinstance(context_only, bool):
        raise TypeError("context_only must be boolean")
    if context_only and data.context_mask is not None and not bool(data.context_mask.any()):
        raise ValueError("context_only fitting requires at least one available context row")
    if context_only and (context is None or context is BASE_CONTEXT):
        raise ValueError("context_only fitting requires active external context")
    before_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    before_mode = model.training
    try:
        if context_only:
            for name, p in model.named_parameters():
                p.requires_grad_(
                    before_flags[name]
                    and (
                        ".context_net." in name
                        or "._proj." in name
                        or name.endswith(".delta_scale")
                    )
                )
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("no trainable parameters for supervised fitting")
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            betas=(model.config.adam_beta1, model.config.adam_beta2),
            eps=model.config.adam_eps,
        )
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = -score_fn(model, observations, labels, context=context, reduce=True) / count
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite supervised loss")
            loss.backward()
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters
            ):
                raise FloatingPointError("non-finite supervised gradient")
            if model.config.supervised_grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters, model.config.supervised_grad_clip_norm, error_if_nonfinite=True
                )
            optimizer.step()
            if not context_only and model.dist.duration.tail_end_probability is not None:
                model.dist.duration._project_tail_parameter()
        model.eval()
        with torch.no_grad():
            final = score_fn(model, observations, labels, context=context, reduce=True)
        if not bool(torch.isfinite(final)):
            raise FloatingPointError("non-finite final supervised score")
        return {
            "objective": objective,
            "updates": steps,
            "context_only": context_only,
            "rows": count,
            "initial_nll_per_row": -float(initial.detach()) / count,
            "final_nll_per_row": -float(final) / count,
        }
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(before_flags[name])
        model.train(before_mode)
