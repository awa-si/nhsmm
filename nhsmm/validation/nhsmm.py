from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from .context_evidence import ContextEvidence, ContextEvidenceConfig, evaluate_context_replication
from .split_fit import SplitFitEvidence, split_fit_context_evidence


def _model_device_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    bias = getattr(model, "duration_logits_bias", None)
    if isinstance(bias, torch.Tensor):
        return bias.device, bias.dtype
    try:
        param = next(model.parameters())
    except (AttributeError, StopIteration, TypeError) as exc:
        raise TypeError("model must expose torch parameters or duration_logits_bias") from exc
    return param.device, param.dtype


def _context_tensor(model: Any, context: Any) -> torch.Tensor:
    device, dtype = _model_device_dtype(model)
    value = torch.as_tensor(context, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim != 1:
        raise ValueError("a single evidence context must be scalar or one-dimensional")
    if not torch.isfinite(value).all():
        raise ValueError("evidence context must contain only finite values")
    expected = getattr(model, "context_dim", None)
    if expected is not None and value.shape[0] != int(expected):
        raise ValueError(
            f"context dimension mismatch: got {value.shape[0]}, expected {int(expected)}"
        )
    return value.reshape(1, 1, -1)


def transition_context_matrix(model: Any, context: Any) -> np.ndarray:
    """Return the effective KxK boundary-transition matrix at one context value.

    For duration-dependent NHSMM transitions, this integrates the transition law over
    the model's current duration distribution, including ``duration_logits_bias``.
    The function is inference-only and restores the model's previous training mode.
    """
    try:
        duration = model.dist.duration
        transition = model.dist.transition
    except AttributeError as exc:
        raise TypeError("model must expose dist.duration and dist.transition") from exc

    ctx = _context_tensor(model, context)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            transition_p = transition.log_matrix(context=ctx, T=1).exp()
            if transition_p.ndim == 5:
                bias = getattr(model, "duration_logits_bias", None)
                duration_p = duration.log_matrix(
                    context=ctx,
                    T=1,
                    soft_dmax=bias,
                ).exp()
                if duration_p.ndim != 4:
                    raise ValueError("duration log_matrix must return [B,T,K,D]")
                if duration_p.shape[:4] != transition_p.shape[:4]:
                    raise ValueError(
                        "duration and transition outputs must agree on [B,T,K,D]"
                    )
                matrix = (duration_p.unsqueeze(-1) * transition_p).sum(dim=-2)[0, 0]
            elif transition_p.ndim == 4:
                matrix = transition_p[0, 0]
            else:
                raise ValueError(
                    "transition log_matrix must return [B,T,K,K] or [B,T,K,D,K]"
                )
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

    out = matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("effective transition matrix must be square [K,K]")
    if not np.isfinite(out).all():
        raise ValueError("effective transition matrix contains non-finite values")
    rows = out.sum(axis=1)
    if not np.allclose(rows, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("effective transition matrix rows must sum to one")
    return out


def evaluate_transition_context_replication(
    model_a: Any,
    model_b: Any,
    contexts: Iterable[Any],
    *,
    config: ContextEvidenceConfig,
) -> ContextEvidence:
    """Evaluate replicated NHSMM transition-context effects across two fitted models."""
    return evaluate_context_replication(
        model_a,
        model_b,
        contexts,
        effect_matrix=transition_context_matrix,
        config=config,
    )


def split_fit_transition_context_evidence(
    observations: Any,
    context: Any,
    *,
    fit_model: Callable[[Any, Any, int], Any],
    evidence_contexts: Iterable[Any],
    config: ContextEvidenceConfig,
    split_index: Optional[int] = None,
) -> SplitFitEvidence:
    """Fit two disjoint NHSMM replicas and validate transition-context replication.

    ``fit_model`` owns all training policy and returns a fitted NHSMM-like model for
    each replica. Validation does not call or modify ``NHSMM.optimize``.
    """
    return split_fit_context_evidence(
        observations,
        context,
        fit_model=fit_model,
        effect_matrix=transition_context_matrix,
        evidence_contexts=evidence_contexts,
        config=config,
        split_index=split_index,
    )
