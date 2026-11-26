"""Native first-order likelihood and VJP for one CPU HSMM."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch

from nhsmm.context import context_options
from nhsmm.filtering import _center_emission_log_probs, _duration_log_hazard_from_log_p
from nhsmm.models.native_errors import NativeInputUnsupported


def load_kernel(path):
    library = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    fn = library.hsmm_forward_backward
    fn.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 11
    fn.restype = ctypes.c_int
    return library, fn


class _RustHSMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pi, em, a, end, cont, fn):
        arrays = [np.ascontiguousarray(v.detach().numpy()) for v in (pi, em, a, end, cont)]
        t, k = arrays[1].shape
        d = arrays[3].shape[-1]
        score = np.empty(1, dtype=np.float64)
        grads = [np.empty_like(v) for v in arrays]
        code = fn(t, k, d, *(ctypes.c_void_p(v.ctypes.data) for v in (*arrays, score, *grads)))
        if code:
            raise ValueError(f"Rust HSMM kernel status {code}")
        ctx.save_for_backward(*(torch.from_numpy(v) for v in grads))
        return pi.new_tensor(score[0])

    @staticmethod
    def backward(ctx, upstream):
        if torch.is_grad_enabled():
            raise RuntimeError("Rust HSMM supports first-order gradients only")
        return (*(upstream * g for g in ctx.saved_tensors), None)


def native_hsmm_likelihood(model, seq, temperature, fn):
    cfg = model.config
    if (
        cfg.model_mode not in {"hsmm", "nhsmm"}
        or not cfg.causal
        or cfg.max_duration < 2
        or len(seq.sequences) != 1
        or len(seq.lengths) != 1
    ):
        raise NativeInputUnsupported("native Rust HSMM requires one causal D>=2 sequence")
    t = int(seq.lengths[0])
    em = seq.log_probs[0]
    if (
        t < 1
        or em.shape != (t, cfg.n_states)
        or em.dtype != torch.float64
        or em.device.type != "cpu"
    ):
        raise NativeInputUnsupported("native Rust HSMM requires one unpadded CPU Float64 sequence")
    kw = dict(temperature=temperature, T=t, soft_dmax=model.duration_logits_bias)
    mask = seq.context_mask
    pi = model.dist.initial.log_matrix(
        context=model._active_context(seq.canonical),
        temperature=temperature,
        T=t,
        **context_options(mask, canonical=True),
    )[0, 0]
    q = model.dist.duration.log_matrix(
        context=model._active_context(seq.contexts), **kw, **context_options(mask, canonical=False)
    )[0]
    a = model.dist.transition.log_matrix(
        context=model._active_context(seq.contexts), **kw, **context_options(mask, canonical=False)
    )[0]
    if pi.shape != (cfg.n_states,) or q.shape != (t, cfg.n_states, cfg.max_duration):
        raise NativeInputUnsupported(
            "native Rust HSMM requires initial [K] and duration [T,K,D] laws"
        )
    hfn = getattr(model.dist.duration, "tail_probability", None)
    h = None
    if callable(hfn):
        h = (
            hfn(
                context=model._active_context(seq.contexts),
                **context_options(mask, canonical=False),
            )
            if getattr(model.dist.duration, "tail_context_enabled", False)
            else hfn()
        )
        if h is not None:
            if torch.is_grad_enabled() and bool((h == 1).any()):
                raise NativeInputUnsupported("native Rust HSMM excludes exact h=1 boundary")
            if h.ndim == 1:
                h = h[None].expand(t, -1)
            else:
                h = h[0]
    end, cont = _duration_log_hazard_from_log_p(q, h)
    d = end.shape[-1]
    if end.shape != (t, cfg.n_states, d) or cont.shape != end.shape or d != cfg.max_duration:
        raise NativeInputUnsupported("native Rust HSMM requires end/continuation [T,K,D] laws")
    if a.ndim == 3:
        a = a.unsqueeze(2).expand(t, cfg.n_states, d, cfg.n_states)
    if a.shape != (t, cfg.n_states, d, cfg.n_states):
        raise NativeInputUnsupported("unsupported transition shape")
    centered, offset = _center_emission_log_probs(em)
    tensors = (pi, centered, a, end, cont)
    if any(
        v.dtype != torch.float64
        or v.device.type != "cpu"
        or bool(torch.isnan(v).any())
        or bool(torch.isposinf(v).any())
        for v in tensors
    ):
        raise ValueError("native Rust HSMM requires CPU Float64 log scores without NaN/+inf")
    return (_RustHSMM.apply(*tensors, fn) + offset.sum()).unsqueeze(0)
