"""Fully observed causal state paths, with a right-censored final episode.

doc:nhsmm.supervised-path (docs/supervised.md)
"""

from __future__ import annotations

import math

import torch

from nhsmm.context import BASE_CONTEXT
from nhsmm.filtering import _duration_log_hazard_from_log_p


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
        model.dist.initial.log_matrix(context=active(data.canonical)), B, "initial"
    )
    q = model._expand_batch(
        model.dist.duration.log_matrix(
            context=active(data.contexts), T=T, soft_dmax=model.duration_logits_bias
        ),
        B,
        "duration",
    )
    transition = model._expand_batch(
        model.dist.transition.log_matrix(context=active(data.contexts), T=T), B, "transition"
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


def fit_supervised(
    model, observations, labels, *, context=None, steps=200, lr=0.03, context_only=False
):
    """Bounded full-batch Adam on observed joint paths; never calls latent fit.

    Uses current parameters, no hidden restarts, no validation or label matching.
    Returns a separate summary and leaves the latent optimization API untouched.
    """
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
    initial = supervised_log_likelihood(model, observations, labels, context=context, reduce=True)
    if not bool(torch.isfinite(initial)):
        raise ValueError("supervised path has zero or non-finite probability")
    if not isinstance(context_only, bool):
        raise TypeError("context_only must be boolean")
    if context_only and (context is None or context is BASE_CONTEXT):
        raise ValueError("context_only fitting requires active external context")
    before_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    if context_only:
        for name, p in model.named_parameters():
            p.requires_grad_(
                before_flags[name]
                and (".context_net." in name or "._proj." in name or name.endswith(".delta_scale"))
            )
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        for name, p in model.named_parameters():
            p.requires_grad_(before_flags[name])
        raise ValueError("no trainable parameters for supervised fitting")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    before_mode = model.training
    model.train()
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (
                -supervised_log_likelihood(
                    model, observations, labels, context=context, reduce=True
                )
                / count
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite supervised loss")
            loss.backward()
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters
            ):
                raise FloatingPointError("non-finite supervised gradient")
            optimizer.step()
            if not context_only and model.dist.duration.tail_end_probability is not None:
                model.dist.duration._project_tail_parameter()
        model.eval()
        with torch.no_grad():
            final = supervised_log_likelihood(
                model, observations, labels, context=context, reduce=True
            )
        if not bool(torch.isfinite(final)):
            raise FloatingPointError("non-finite final supervised score")
        return {
            "objective": "observed_joint_causal_prefix",
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
