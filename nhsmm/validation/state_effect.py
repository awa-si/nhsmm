from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from .alignment import StateAlignment, align_state_centers, emission_centers
from .context_evidence import ContextEvidence, ContextEvidenceConfig
from .nhsmm import _context_tensor
from .split_fit import SplitFitEvidence, _slice


def _reorder_state_axis(value: Any, alignment: StateAlignment) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    k = len(alignment.candidate_to_reference)
    if array.ndim < 1 or array.shape[0] != k:
        raise ValueError(f"state effect tensor must have first dimension K={k}")
    inverse = np.asarray(alignment.reference_to_candidate, dtype=np.int64)
    return array[inverse]


def _effect_vector(tensors: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if len(tensors) < 2:
        raise ValueError("at least two context grid points are required")
    shape = tensors[0].shape
    if not shape or shape[0] == 0:
        raise ValueError("state effect tensors must contain at least one state")
    for tensor in tensors:
        if tensor.shape != shape:
            raise ValueError("state effect tensors must have equal shape across contexts")
        if not np.isfinite(tensor).all():
            raise ValueError("state effect tensors must contain only finite values")
    values = np.stack(tensors, axis=0)
    flat = values.reshape(values.shape[0], -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    amplitude = float(np.mean(np.ptp(flat, axis=0)))
    return centered.reshape(-1), amplitude


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_state_context_replication(
    model_a: Any,
    model_b: Any,
    contexts: Iterable[Any],
    *,
    effect_tensor: Callable[[Any, Any], Any],
    config: ContextEvidenceConfig,
    centers: Callable[[Any], Any] = emission_centers,
) -> ContextEvidence:
    """Evaluate replication of any context effect whose first axis is latent state.

    ``effect_tensor(model, context)`` returns ``[K, ...]``. Only the first axis is
    interpreted by validation; all remaining axes are component semantics owned by
    the adapter (duration bins, observation features, or another state-conditioned
    quantity). Existing transition-specific validation remains separate because it
    has two latent-state axes and conditional-change normalization semantics.
    """
    grid = list(contexts)
    if len(grid) < 2:
        raise ValueError("contexts must contain at least two grid points")

    alignment = align_state_centers(centers(model_a), centers(model_b))
    effects_a: list[np.ndarray] = []
    effects_b: list[np.ndarray] = []
    for context in grid:
        a = np.asarray(effect_tensor(model_a, context), dtype=np.float64)
        b = _reorder_state_axis(effect_tensor(model_b, context), alignment)
        if a.shape != b.shape:
            raise ValueError("replica state effect tensors must have equal shape")
        effects_a.append(a)
        effects_b.append(b)

    vec_a, amp_a = _effect_vector(effects_a)
    vec_b, amp_b = _effect_vector(effects_b)
    replication_corr = _corr(vec_a, vec_b)
    nz = (np.abs(vec_a) > 1e-4) & (np.abs(vec_b) > 1e-4)
    agreement = float(((vec_a[nz] * vec_b[nz]) > 0.0).mean()) if np.any(nz) else 0.0
    amp_min = min(amp_a, amp_b)
    selected = (
        replication_corr >= config.replication_corr_min
        and amp_min >= config.min_replica_amplitude
    )
    return ContextEvidence(
        selected=bool(selected),
        replication_corr=replication_corr,
        direction_agreement=agreement,
        replica_amplitudes=(amp_a, amp_b),
        min_replica_amplitude=amp_min,
        state_alignment=alignment,
        n_contexts=len(grid),
    )


def split_fit_state_context_evidence(
    observations: Any,
    context: Any,
    *,
    fit_model: Callable[[Any, Any, int], Any],
    effect_tensor: Callable[[Any, Any], Any],
    evidence_contexts: Iterable[Any],
    config: ContextEvidenceConfig,
    centers: Callable[[Any], Any] = emission_centers,
    split_index: Optional[int] = None,
) -> SplitFitEvidence:
    """Split-fit wrapper for arbitrary state-indexed context effects."""
    n = len(observations)
    if len(context) != n:
        raise ValueError("observations and context must have equal first-axis length")
    if n < 2:
        raise ValueError("at least two independent training units are required")
    split = n // 2 if split_index is None else int(split_index)
    if split <= 0 or split >= n:
        raise ValueError("split_index must leave both replicas non-empty")

    model_a = fit_model(_slice(observations, 0, split), _slice(context, 0, split), 0)
    model_b = fit_model(_slice(observations, split, n), _slice(context, split, n), 1)
    evidence = evaluate_state_context_replication(
        model_a,
        model_b,
        evidence_contexts,
        effect_tensor=effect_tensor,
        config=config,
        centers=centers,
    )
    return SplitFitEvidence(evidence=evidence, split_index=split, replica_sizes=(split, n - split))


def duration_context_tensor(model: Any, context: Any) -> np.ndarray:
    """Return the effective duration distribution ``[K,D]`` at one context."""
    try:
        duration = model.dist.duration
    except AttributeError as exc:
        raise TypeError("model must expose dist.duration") from exc
    ctx = _context_tensor(model, context)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            value = duration.log_matrix(
                context=ctx,
                T=1,
                soft_dmax=getattr(model, "duration_logits_bias", None),
            ).exp()
            if value.ndim != 4:
                raise ValueError("duration log_matrix must return [B,T,K,D]")
            if value.shape[0] != 1 or value.shape[1] != 1:
                raise ValueError("duration evidence evaluation expects B=T=1")
            if value.shape[2] <= 0 or value.shape[3] <= 0:
                raise ValueError("duration log_matrix must contain positive K and D dimensions")
            out = value[0, 0].detach().cpu().numpy().astype(np.float64, copy=False)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    if not np.isfinite(out).all():
        raise ValueError("duration context tensor contains non-finite values")
    if not np.allclose(out.sum(axis=-1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("duration context tensor rows must sum to one")
    return out


def emission_context_tensor(model: Any, context: Any) -> np.ndarray:
    """Return state-conditioned emission means ``[K,F]`` at one context."""
    try:
        emission = model.dist.emission
    except AttributeError as exc:
        raise TypeError("model must expose dist.emission") from exc
    ctx = _context_tensor(model, context)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            dist = emission.forward(context=ctx, return_dist=True)
            value = dist.mean
            while value.ndim > 2 and value.shape[0] == 1:
                value = value[0]
            out = value.detach().cpu().numpy().astype(np.float64, copy=False)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    if out.ndim != 2 or out.shape[0] <= 0 or out.shape[1] <= 0:
        raise ValueError("emission context tensor must have non-empty shape [K,F]")
    if not np.isfinite(out).all():
        raise ValueError("emission context tensor contains non-finite values")
    return out


def initial_context_tensor(model: Any, context: Any) -> np.ndarray:
    """Return the context-conditioned initial-state probabilities ``[K]``."""
    try:
        initial = model.dist.initial
    except AttributeError as exc:
        raise TypeError("model must expose dist.initial") from exc
    ctx = _context_tensor(model, context)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            value = initial.expected_probs(context=ctx)
            while value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            out = value.detach().cpu().numpy().astype(np.float64, copy=False)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    if out.ndim != 1 or out.shape[0] <= 0:
        raise ValueError("initial context tensor must have non-empty shape [K]")
    if not np.isfinite(out).all():
        raise ValueError("initial context tensor contains non-finite values")
    if np.any(out < 0.0):
        raise ValueError("initial context tensor must contain non-negative probabilities")
    if not np.isclose(out.sum(), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("initial context tensor must sum to one")
    return out


def evaluate_initial_context_replication(model_a: Any, model_b: Any, contexts: Iterable[Any], *, config: ContextEvidenceConfig) -> ContextEvidence:
    return evaluate_state_context_replication(model_a, model_b, contexts, effect_tensor=initial_context_tensor, config=config)


def evaluate_duration_context_replication(model_a: Any, model_b: Any, contexts: Iterable[Any], *, config: ContextEvidenceConfig) -> ContextEvidence:
    return evaluate_state_context_replication(model_a, model_b, contexts, effect_tensor=duration_context_tensor, config=config)


def evaluate_emission_context_replication(model_a: Any, model_b: Any, contexts: Iterable[Any], *, config: ContextEvidenceConfig) -> ContextEvidence:
    return evaluate_state_context_replication(model_a, model_b, contexts, effect_tensor=emission_context_tensor, config=config)


def split_fit_initial_context_evidence(observations: Any, context: Any, *, fit_model: Callable[[Any, Any, int], Any], evidence_contexts: Iterable[Any], config: ContextEvidenceConfig, split_index: Optional[int] = None) -> SplitFitEvidence:
    return split_fit_state_context_evidence(observations, context, fit_model=fit_model, effect_tensor=initial_context_tensor, evidence_contexts=evidence_contexts, config=config, split_index=split_index)


def split_fit_duration_context_evidence(observations: Any, context: Any, *, fit_model: Callable[[Any, Any, int], Any], evidence_contexts: Iterable[Any], config: ContextEvidenceConfig, split_index: Optional[int] = None) -> SplitFitEvidence:
    return split_fit_state_context_evidence(observations, context, fit_model=fit_model, effect_tensor=duration_context_tensor, evidence_contexts=evidence_contexts, config=config, split_index=split_index)


def split_fit_emission_context_evidence(observations: Any, context: Any, *, fit_model: Callable[[Any, Any, int], Any], evidence_contexts: Iterable[Any], config: ContextEvidenceConfig, split_index: Optional[int] = None) -> SplitFitEvidence:
    return split_fit_state_context_evidence(observations, context, fit_model=fit_model, effect_tensor=emission_context_tensor, evidence_contexts=evidence_contexts, config=config, split_index=split_index)
