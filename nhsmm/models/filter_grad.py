"""Exact reverse-mode derivative of the shared causal state-age recursion."""

import numpy as np
import torch


def _responsibility(log_mass, log_total):
    # Empty support has zero derivative, matching the shared safe log reductions.
    safe_total = np.where(np.isfinite(log_total), log_total, 0)
    return np.exp(log_mass - safe_total)


class _CausalPrefixAdjoint(torch.autograd.Function):
    """doc:nhsmm.causal-prefix-adjoint (docs/supervised.md#hsmm-gradient-cost)"""

    @staticmethod
    def forward(ctx, initial, emission, transition, end, continuation, lengths, tail):
        from nhsmm.models.forward import _causal_prefix_serial

        alpha = _causal_prefix_serial(
            initial,
            emission,
            transition,
            end,
            continuation,
            lengths,
            tail,
            normalized=True,
            direct_reductions=True,
        )
        ctx.save_for_backward(initial, emission, transition, end, continuation, lengths, alpha)
        ctx.tail = tail
        return alpha

    @staticmethod
    def backward(ctx, gradient):
        initial, emission, transition, end, continuation, lengths, alpha = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Saved forward intermediates carry no derivative graph. Recompute
            # using the same owner for exact higher derivatives on explicit request.
            from nhsmm.models.forward import _causal_prefix_serial

            inputs = (initial, emission, transition, end, continuation)
            required = [p for p in inputs if p.requires_grad]
            output = _causal_prefix_serial(*inputs, lengths, ctx.tail, normalized=True)
            grads = iter(
                torch.autograd.grad(
                    output, required, gradient, create_graph=True, allow_unused=True
                )
            )
            return tuple(next(grads) if p.requires_grad else None for p in inputs) + (None, None)

        # Values are CPU Float32/64 by the dispatch guard. NumPy views share
        # input storage; only result/work arrays are written. Derivatives still
        # return through this exact custom VJP into the Torch parameter graph.
        initial, emission, transition, end, continuation, lengths, alpha = (
            value.detach().numpy()
            for value in (initial, emission, transition, end, continuation, lengths, alpha)
        )
        gradient = gradient.detach().numpy()
        B, T, K, D = alpha.shape
        gi, ge, ga, gh, gc = (
            np.zeros_like(value) for value in (initial, emission, transition, end, continuation)
        )
        probability = np.exp(alpha)
        # Compute all edge responsibilities once in native batched operations.
        reset = alpha[:, :-1, :, :, None] + end[:, :-1, :, :, None] + transition[:, :-1]
        reset_total = np.logaddexp.reduce(np.logaddexp.reduce(reset, axis=2), axis=2)
        continue_mass = alpha[:, :-1] + continuation[:, :-1]
        first_total = (
            np.logaddexp(reset_total, continue_mass[..., 0]) if ctx.tail and D == 1 else reset_total
        )
        reset_weight = _responsibility(reset, first_total[:, :, None, None, :])
        continue_weight = np.zeros_like(continue_mass)
        if D > 1:
            continue_weight[..., :-1] = np.isfinite(continue_mass[..., :-1])
        if ctx.tail:
            if D == 1:
                last_total = first_total
            else:
                last_total = np.logaddexp(continue_mass[..., -2], continue_mass[..., -1])
                continue_weight[..., -2] = _responsibility(continue_mass[..., -2], last_total)
            continue_weight[..., -1] = _responsibility(continue_mass[..., -1], last_total)
        destination = np.minimum(np.arange(D) + 1, D - 1)
        carry = np.zeros_like(alpha[:, 0])
        age_independent = transition.shape[3] == 1 and D > 1
        for t in range(T - 1, -1, -1):
            adjoint = gradient[:, t] + carry
            g = adjoint - probability[:, t] * adjoint.sum(axis=(1, 2), keepdims=True)
            g = np.where((lengths > t)[:, None, None], g, 0)
            ge[:, t] = g.sum(axis=-1)
            if t == 0:
                gi[...] = g[..., 0]
                break
            reset_gradient = reset_weight[:, t - 1] * g[..., 0][:, None, None, :]
            carry = reset_gradient.sum(axis=-1)
            gh[:, t - 1] = carry
            ga[:, t - 1] = (
                reset_gradient.sum(axis=2, keepdims=True) if age_independent else reset_gradient
            )
            continue_gradient = continue_weight[:, t - 1] * g[..., destination]
            gc[:, t - 1] = continue_gradient
            carry += continue_gradient
        return tuple(torch.from_numpy(value) for value in (gi, ge, ga, gh, gc)) + (None, None)
