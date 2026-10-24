"""Explicit stationary state/age entry law for time-homogeneous causal models."""

import torch

from nhsmm.filtering import HSMMFilterState


@torch.inference_mode()
def stationary_filter_prior(model, *, temperature=None):
    """Unconditional equilibrium prior, not a posterior or a learned initial pi.

    Episode-entry equilibrium uses the duration-averaged destination kernel.
    Age occupancy is entry mass times survival; D+ visits divide by tail risk.
    """
    from nhsmm.distributions.default import Duration, Transition

    if model.config.model_mode not in {"hmm", "hsmm"} or not model.config.causal:
        raise ValueError("stationary prior requires a static causal HMM or HSMM")
    if model.dist is None:
        raise RuntimeError("initialize model distributions first")
    if type(model.dist.duration) is not Duration or type(model.dist.transition) is not Transition:
        raise ValueError("stationary prior requires canonical static distributions")
    log_q = model.dist.duration.log_matrix(
        T=1, temperature=temperature, soft_dmax=model.duration_logits_bias
    )[0, 0].double()
    log_a = model.dist.transition.log_matrix(T=1, temperature=temperature)[0, 0].double()
    if log_a.ndim == 2:
        log_a = log_a[:, None, :].expand(-1, log_q.shape[-1], -1)
    normalized_q = log_q.log_softmax(-1)
    q = normalized_q.exp()
    a = log_a.softmax(-1)
    kernel = (q[..., None] * a).sum(1)
    k = kernel.shape[0]
    matrix = kernel.T - torch.eye(k, dtype=kernel.dtype, device=kernel.device)
    matrix = matrix.clone()
    matrix[-1] = 1
    rhs = kernel.new_zeros(k)
    rhs[-1] = 1
    if int(torch.linalg.matrix_rank(matrix)) != k:
        raise ValueError("episode-entry kernel has no unique stationary law")
    entry = torch.linalg.solve(matrix, rhs)
    tolerance = 64 * torch.finfo(entry.dtype).eps
    if not bool(torch.isfinite(entry).all()) or bool((entry < -tolerance).any()):
        raise ValueError("invalid stationary episode-entry law")
    entry = entry.clamp_min(0)
    entry /= entry.sum()
    if not torch.allclose(entry @ kernel, entry, atol=tolerance, rtol=0):
        raise ValueError("stationary episode-entry residual exceeds numerical tolerance")
    survival = torch.logcumsumexp(normalized_q.flip(-1), dim=-1).flip(-1)
    occupancy = entry.log()[:, None] + survival
    tail = model.dist.duration.tail_probability()
    if tail is not None:
        h = tail.double()
        if not bool(torch.isfinite(h).all()) or bool(((h <= 0) | (h > 1)).any()):
            raise ValueError("invalid geometric tail risk")
        occupancy[:, -1] -= h.log()
    value = occupancy.flatten().log_softmax(0).reshape(1, *occupancy.shape)
    value = value.to(model.duration_logits_bias)
    value = value.flatten(1).log_softmax(-1).reshape_as(value)
    return HSMMFilterState(value)
