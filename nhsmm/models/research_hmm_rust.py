"""Opt-in, research-only Rust likelihood for one static finite CPU HMM.

The ABI intentionally requires a user-built library; it is not bundled in
NHSMM wheels and is not an inference/production backend.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import _center_emission_log_probs
from nhsmm.models.research_rust import NativeInputUnsupported


def load_kernel(path: str | Path):
    resolved = Path(path).expanduser().resolve(strict=True)
    library = ctypes.CDLL(str(resolved))
    function = library.hmm_forward_backward
    function.argtypes = [ctypes.c_size_t, ctypes.c_size_t] + [ctypes.c_void_p] * 7
    function.restype = ctypes.c_int
    # Keep the library alive for the lifetime of the function pointer.
    return library, function


class _RustHMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial, transition, emission, function):
        arrays = [np.ascontiguousarray(t.detach().numpy()) for t in (initial, transition, emission)]
        likelihood = np.empty(1, dtype=np.float64)
        grads = [np.empty_like(a) for a in arrays]
        status = function(
            len(arrays[2]),
            len(arrays[0]),
            *(ctypes.c_void_p(a.ctypes.data) for a in (*arrays, likelihood, *grads)),
        )
        if status != 0:
            raise ValueError(f"research Rust HMM kernel failed with status {status}")
        ctx.save_for_backward(*(torch.from_numpy(g) for g in grads))
        return initial.new_tensor(likelihood[0])

    @staticmethod
    def backward(ctx, upstream):
        if torch.is_grad_enabled():
            raise RuntimeError("research Rust HMM kernel supports first-order gradients only")
        return (*(upstream * g for g in ctx.saved_tensors), None)


def research_hmm_likelihood(model, sequence, temperature, function):
    """Fail closed outside the measured single-sequence Float64 D=1 scope."""
    config = model.config
    if (
        config.model_mode != "hmm"
        or not config.causal
        or config.duration_tail
        or config.context_dim is not None
        or config.use_context_encoder
        or config.transition_type != "ergodic"
        or not config.allow_self_transitions
    ):
        raise NativeInputUnsupported("research Rust HMM requires a static causal ergodic D=1 model")
    if len(sequence.sequences) != 1 or len(sequence.lengths) != 1:
        raise NativeInputUnsupported("research Rust HMM requires exactly one sequence")
    emission = sequence.log_probs[0]
    t, k = emission.shape
    if t < 1 or int(sequence.lengths[0]) != t or k != config.n_states:
        raise NativeInputUnsupported("research Rust HMM requires one nonempty unpadded sequence")
    if emission.device.type != "cpu" or emission.dtype != torch.float64:
        raise NativeInputUnsupported("research Rust HMM requires CPU Float64 emissions")

    initial_law = model.dist.initial.log_matrix(temperature=temperature, T=t)
    transition_law = model.dist.transition.log_matrix(temperature=temperature, T=t)
    duration = model.dist.duration.log_matrix(
        temperature=temperature, T=t, soft_dmax=model.duration_logits_bias
    )
    if (
        initial_law.shape != (1, 1, k)
        or transition_law.shape != (1, t, k, 1, k)
        or duration.shape != (1, t, k, 1)
        or not bool(
            torch.equal(transition_law[:, 1:], transition_law[:, :1].expand(-1, t - 1, -1, -1, -1))
        )
        or not bool(torch.all(duration == 0))
    ):
        raise NativeInputUnsupported("research Rust HMM requires static D=1 laws")
    initial = initial_law[0, 0]
    transition = transition_law[0, 0, :, 0, :]
    if any(
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float64
        or not bool(torch.isfinite(tensor).all())
        for tensor in (initial, transition, emission)
    ):
        raise ValueError("research Rust HMM requires finite CPU Float64 log scores")
    # No transition is traversed for T=1; match the ordinary path's absent gradient.
    if t == 1:
        transition = transition.detach()
    # Match the ordinary causal likelihood's per-row emission centering.
    centered, row_offsets = _center_emission_log_probs(emission)
    # D=1 duration has probability one. Keep its zero-gradient graph link.
    return (
        _RustHMM.apply(initial, transition, centered, function)
        + row_offsets.sum()
        + duration.sum() * 0.0
    ).unsqueeze(0)
