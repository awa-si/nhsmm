"""Exact first-order tail-score derivative at the finite compatibility boundary.

doc:nhsmm.tail-boundary-gradient (docs/duration-tail-contract.md)

At h=1 the log semiring omits zero-weight continuation paths. Their one-sided
derivative is -1 times the weight with that single continuation edge removed.
Forward/backward messages evaluate that missing flux without taking log(0)
derivatives or approximating h. Only score evaluation needs this correction;
the ordinary forward values and all interior derivatives remain unchanged.
"""

from nhsmm.context import context_options

import torch

from nhsmm.filtering import (
    _center_emission_log_probs,
    _duration_log_hazard_from_log_p,
    _safe_logaddexp,
    _safe_logsumexp,
)


class _TailBoundaryScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score, tail, missing_flux):
        ctx.save_for_backward(missing_flux)
        return score.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("tail boundary score supports first-order derivatives only")
        (flux,) = ctx.saved_tensors
        return grad_output, -(grad_output[:, None] * flux).sum(dim=0), None


def tail_boundary_score(model, sequence, scores, temperature=None):
    tail_fn = getattr(model.dist.duration, "tail_probability", None)
    tail = tail_fn() if callable(tail_fn) else None
    if (
        tail is None
        or not torch.is_grad_enabled()
        or not tail.requires_grad
        or not bool((tail == 1).any())
    ):
        return scores
    # The boundary VJP is explicitly first-order; no silently incorrect Hessian.
    with torch.no_grad():
        flux = _missing_flux(model, sequence, scores, tail, temperature)
    return _TailBoundaryScore.apply(scores, tail, flux.to(tail))


def _missing_flux(model, sequence, scores, tail, temperature):
    B, T, K = sequence.log_probs.shape
    D = model.dist.duration.max_duration
    result = tail.new_zeros((B, K), dtype=torch.float64)
    if T < 2:
        return result
    active_context = model._active_context
    pi = model.dist.initial.log_matrix(
        context=active_context(sequence.canonical),
        temperature=temperature,
        T=T,
        **context_options(sequence.context_mask, canonical=True),
    )
    q = model.dist.duration.log_matrix(
        context=active_context(sequence.contexts),
        temperature=temperature,
        T=T,
        soft_dmax=model.duration_logits_bias,
        **context_options(sequence.context_mask, canonical=False),
    )
    A = model.dist.transition.log_matrix(
        context=active_context(sequence.contexts),
        temperature=temperature,
        T=T,
        soft_dmax=model.duration_logits_bias,
        **context_options(sequence.context_mask, canonical=False),
    )
    pi = model._expand_batch(pi, B, "initial logits").double()
    q = model._expand_batch(q, B, "duration logits").double()
    A = model._expand_batch(A, B, "transition logits").double()
    if model.dist.transition.max_duration is None:
        A = A.unsqueeze(3).expand(B, T, K, D, K)
    h = tail.double()
    if model.config.causal:
        ends, cont = _duration_log_hazard_from_log_p(
            q.reshape(B * T, K, D), h.view(1, K).expand(B * T, K)
        )
        ends, cont = ends.reshape_as(q), cont.reshape_as(q)
    else:
        # Open segment automaton: attach q only when completing the segment.
        ends = q.clone()
        ends[..., -1] += h.log()
        cont = torch.zeros_like(q)
        cont[..., -1] = torch.where(h < 1, torch.log1p(-h), -torch.inf)
    emissions = sequence.log_probs.double()
    for b, length in enumerate(sequence.lengths.tolist()):
        if length < 2 or not bool(torch.isfinite(scores[b])):
            continue
        frontier = q.new_full((K, D), -torch.inf)
        frontier[:, 0] = pi[b, 0] + emissions[b, 0]
        prefixes = [frontier]
        for t in range(1, length):
            boundary = _safe_logsumexp(
                (frontier + ends[b, t - 1]).unsqueeze(-1) + A[b, t - 1], dim=(0, 1)
            )
            if D == 1:
                nxt = _safe_logaddexp(boundary, frontier[:, -1] + cont[b, t - 1, :, -1])[:, None]
            else:
                continuation = frontier[:, :-1] + cont[b, t - 1, :, :-1]
                last = _safe_logaddexp(continuation[:, -1], frontier[:, -1] + cont[b, t - 1, :, -1])
                nxt = torch.cat((boundary[:, None], continuation[:, :-1], last[:, None]), -1)
            frontier = nxt + emissions[b, t, :, None]
            prefixes.append(frontier)
        backward = torch.zeros_like(frontier) if model.config.causal else ends[b, length - 1]
        total = _safe_logsumexp((frontier + backward).flatten(), dim=0)
        for t in range(length - 2, -1, -1):
            # Derivative of (1-h) at h=1 is -1, including zero-mass states.
            contribution = (
                prefixes[t][:, -1] + emissions[b, t + 1] + backward[:, -1] - total
            ).exp()
            result[b] += torch.where(h == 1, contribution, 0.0)
            future = emissions[b, t + 1, :, None] + backward
            boundary = ends[b, t] + _safe_logsumexp(A[b, t] + future[:, 0].view(1, 1, K), dim=-1)
            continuation = cont[b, t] + torch.cat((future[:, 1:], future[:, -1:]), -1)
            backward = _safe_logaddexp(boundary, continuation)
    return result


def tail_boundary_filter_score(model, paths, scores, inputs):
    """Restore zero-continuation derivatives of prefix state log scores.

    The log recursion retains positive paths. At h=1 a continuation path has
    zero value but nonzero one-sided derivative. Propagate the positive flux
    obtained by removing each such edge, normalized by the original prefix
    evidence. Subtract its total-evidence contribution from the labelled-state
    contribution. This is a first-order boundary VJP, not a perturbed hazard.
    """
    tail_fn = getattr(model.dist.duration, "tail_probability", None)
    tail = tail_fn() if callable(tail_fn) else None
    if (
        tail is None
        or not torch.is_grad_enabled()
        or not tail.requires_grad
        or not bool((tail == 1).any())
    ):
        return scores
    with torch.no_grad():
        flux = _missing_filter_flux(paths, scores, tail, inputs)
    return _TailBoundaryScore.apply(scores, tail, flux.to(tail))


def _missing_filter_flux(paths, scores, tail, inputs):
    from nhsmm.filtering import _causal_filter_step, _initialize_causal_frontier

    pi, emission, A, end, continuation = (value.double() for value in inputs)
    emission, _ = _center_emission_log_probs(emission)
    B, _, K, D = end.shape
    targets = torch.nonzero(tail == 1, as_tuple=False).flatten()
    result = emission.new_zeros((B, K))
    for b, labels in enumerate(paths):
        if len(labels) < 2 or not bool(torch.isfinite(scores[b])):
            continue
        frontier, _ = _initialize_causal_frontier(pi[b : b + 1], emission[b, 0:1], D)
        frontier = frontier[0]
        extra = emission.new_full((len(targets), K, D), -torch.inf)
        for t in range(1, len(labels)):
            following, log_z = _causal_filter_step(
                frontier[None],
                emission[b, t : t + 1],
                end[b, t - 1 : t],
                continuation[b, t - 1 : t],
                A[b, t - 1 : t],
                tail_enabled=True,
            )
            reset = _safe_logsumexp(
                (extra + end[b, t - 1]).unsqueeze(-1) + A[b, t - 1],
                dim=(1, 2),
            )
            cont = extra + continuation[b, t - 1]
            if D == 1:
                predicted = _safe_logaddexp(reset, cont[..., -1]).unsqueeze(-1)
            else:
                last = _safe_logaddexp(cont[..., -2], cont[..., -1])
                predicted = torch.cat((reset[..., None], cont[..., :-2], last[..., None]), -1)
            # The derivative of the omitted (1-h) edge is -1. Store its
            # positive missing mass here; the VJP applies the minus sign.
            index = torch.arange(len(targets), device=targets.device)
            predicted[index, targets, -1] = _safe_logaddexp(
                predicted[index, targets, -1], frontier[targets, -1]
            )
            extra = predicted + emission[b, t, :, None] - log_z[0]
            frontier = following[0]
            label = labels[t]
            log_label_mass = _safe_logsumexp(frontier[label], dim=-1)
            label_flux = _safe_logsumexp(extra[:, label], dim=-1) - log_label_mass
            total_flux = _safe_logsumexp(extra, dim=(1, 2))
            result[b, targets] += label_flux.exp() - total_flux.exp()
    if not torch.isfinite(result).all():
        raise RuntimeError("tail boundary filtering derivative is non-finite")
    return result
