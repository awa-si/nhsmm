"""Causal output assembly preserves all prefixes and gradients."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from nhsmm import NHSMM, ModelConfig
from nhsmm.filtering import (
    _causal_filter_step,
    _duration_log_hazard_from_log_p,
    _initialize_causal_frontier,
)


def legacy_forward(self, X, router, temperature=None):
    """Unnormalized causal forward recursion over ``(state, episode_age)``.

    Boundary ``t-1 -> t`` uses duration and transition quantities conditioned
    only on ``F_{t-1}``; ``x_t`` enters after propagation through its emission.
    """

    B, T, K = router.log_probs.shape[:3]
    D = int(self.dist.duration.max_duration)
    alpha = router.log_probs.new_full((B, T, K, D), float("-inf"))
    if T == 0:
        return alpha

    initial_logits = self.dist.initial.log_matrix(
        context=self._active_context(router.canonical),
        temperature=temperature,
        T=T,
    )
    duration_logits = self.dist.duration.log_matrix(
        context=self._active_context(router.context),
        temperature=temperature,
        T=T,
        soft_dmax=self.duration_logits_bias,
    )
    transition_logits = self.dist.transition.log_matrix(
        context=self._active_context(router.context),
        temperature=temperature,
        T=T,
        soft_dmax=self.duration_logits_bias,
    )

    initial_logits = self._expand_batch(initial_logits, B, "initial logits")
    duration_logits = self._expand_batch(duration_logits, B, "duration logits")
    transition_logits = self._expand_batch(transition_logits, B, "transition logits")

    expected_initial = (B, 1, K)
    expected_duration = (B, T, K, D)
    if initial_logits.shape != expected_initial:
        raise ValueError(f"initial logits must be {expected_initial}, got {initial_logits.shape}")
    if duration_logits.shape != expected_duration:
        raise ValueError(
            f"duration logits must be {expected_duration}, got {duration_logits.shape}"
        )

    duration_dependent_transition = self.dist.transition.max_duration is not None
    expected_transition = (B, T, K, D, K) if duration_dependent_transition else (B, T, K, K)
    if transition_logits.shape != expected_transition:
        raise ValueError(
            f"transition logits must be {expected_transition}, got {transition_logits.shape}"
        )

    tail_probability_fn = getattr(self.dist.duration, "tail_probability", None)
    tail_end_probability = tail_probability_fn() if callable(tail_probability_fn) else None
    expanded_tail = None
    if tail_end_probability is not None:
        expanded_tail = (
            tail_end_probability.to(device=duration_logits.device, dtype=duration_logits.dtype)
            .view(1, 1, K)
            .expand(B, T, K)
            .reshape(B * T, K)
        )
    log_end, log_continue = _duration_log_hazard_from_log_p(
        duration_logits.reshape(B * T, K, D), expanded_tail
    )
    log_end = log_end.reshape(B, T, K, D)
    log_continue = log_continue.reshape(B, T, K, D)

    # Recurse on a normalized frontier and carry the cumulative evidence
    # separately. This preserves the public unnormalized-alpha contract while
    # preventing avoidable Float32 drift from repeatedly propagating a large
    # sequence-level log offset through logsumexp operations.
    frontier = alpha.new_full((B, K, D), float("-inf"))
    cumulative_log_z = alpha.new_zeros(B)

    valid0 = X.lengths > 0
    normalized_initial, initial_log_z = _initialize_causal_frontier(
        initial_logits[valid0, 0], router.log_probs[valid0, 0], D
    )
    frontier[valid0] = normalized_initial
    cumulative_log_z[valid0] = initial_log_z
    alpha[valid0, 0] = normalized_initial + initial_log_z[:, None, None]

    for t in range(1, T):
        valid = t < X.lengths
        if not bool(valid.any()):
            break
        active = torch.nonzero(valid, as_tuple=False).squeeze(-1)

        transition = transition_logits[active, t - 1]
        if not duration_dependent_transition:
            transition = transition.unsqueeze(2)
        normalized, step_log_z = _causal_filter_step(
            frontier[active],
            router.log_probs[active, t],
            log_end[active, t - 1],
            log_continue[active, t - 1],
            transition,
            tail_enabled=tail_end_probability is not None,
        )
        frontier[active] = normalized
        cumulative_log_z[active] = cumulative_log_z[active] + step_log_z
        alpha[active, t] = normalized + cumulative_log_z[active, None, None]

    return alpha


@pytest.mark.parametrize("lengths", [(7, 7), (7, 4), (7, 0), (0, 0)])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("duration", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("context_dim", [None, 2])
def test_causal_assembled_output_and_gradients_match_legacy(
    lengths, tail, duration, dtype, context_dim
):
    model = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=duration,
            duration_tail=tail,
            duration_tail_init_probability=0.4,
            causal=True,
            use_context_encoder=False,
            context_dim=context_dim,
            dropout=0,
            seed=127,
            verbose=False,
        ),
        device="cpu",
    ).to(dtype=dtype)
    model.initialize_distributions(jitter=0)
    model.eval()
    torch.manual_seed(923)
    steps = max(lengths)
    observations = torch.randn(2, steps, 3, dtype=dtype)
    contexts = torch.randn(2, steps, context_dim or 0, dtype=dtype)
    outcomes = []
    for reference in (True, False):
        model.zero_grad(set_to_none=True)
        emission = observations.clone().requires_grad_(True)
        context = contexts.clone().requires_grad_(True)
        sequence = SimpleNamespace(lengths=torch.tensor(lengths, dtype=torch.long))
        router = SimpleNamespace(
            log_probs=emission,
            canonical=emission.new_zeros((2, 1, context_dim or 0)),
            context=context,
        )
        alpha = (
            legacy_forward(model, sequence, router, temperature=0.71)
            if reference
            else model._forward_causal_hazard(sequence, router, temperature=0.71)
        )
        # Compare all prefix evidence gradients, not only the terminal score.
        valid = torch.arange(steps)[None, :] < sequence.lengths[:, None]
        gradients = None
        if bool(valid.any()):
            loss = torch.logsumexp(alpha[valid].flatten(1), dim=1).sum()
            loss.backward()
            gradients = {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            }
        outcomes.append(
            (
                alpha.detach(),
                None if emission.grad is None else emission.grad.clone(),
                gradients,
                None if context.grad is None else context.grad.clone(),
            )
        )
    expected, actual = outcomes
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    if expected[2] is None:
        assert actual[2] is None
        return
    tolerance = 1e-11 if dtype == torch.float64 else 2e-5
    torch.testing.assert_close(actual[1], expected[1], atol=tolerance, rtol=tolerance)
    if expected[3] is None:
        assert actual[3] is None
    else:
        torch.testing.assert_close(actual[3], expected[3], atol=tolerance, rtol=tolerance)
    assert actual[2].keys() == expected[2].keys()
    for name in actual[2]:
        if expected[2][name] is None:
            assert actual[2][name] is None, name
        else:
            assert torch.isfinite(actual[2][name]).all(), name
            torch.testing.assert_close(
                actual[2][name], expected[2][name], atol=tolerance, rtol=tolerance
            )


@pytest.mark.parametrize(
    "model_dtype, emission_dtype",
    [
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
    ],
)
@pytest.mark.parametrize("tail", [False, True])
def test_mixed_dtype_output_contract_matches_legacy(model_dtype, emission_dtype, tail):
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=2,
            max_duration=3,
            causal=True,
            use_context_encoder=False,
            duration_tail=tail,
            duration_tail_init_probability=0.4,
            dropout=0,
            seed=73,
        ),
        device="cpu",
    ).to(dtype=model_dtype)
    model.initialize_distributions(jitter=0)
    model.eval()
    torch.manual_seed(128)
    values = torch.randn(2, 6, 2, dtype=emission_dtype)
    for lengths in ((0, 0), (6, 3)):
        outcomes = []
        for original in (True, False):
            model.zero_grad(set_to_none=True)
            emission = values.clone()
            sequence = SimpleNamespace(lengths=torch.tensor(lengths, dtype=torch.long))
            router = SimpleNamespace(
                log_probs=emission,
                context=emission.new_empty((2, 6, 0)),
                canonical=emission.new_empty((2, 1, 0)),
            )
            # Mixed finite model/router dtypes are rejected by legacy frontier
            # updates. Preserve that rejection; do not introduce a new dtype API.
            try:
                with torch.no_grad():
                    output = (
                        legacy_forward(model, sequence, router)
                        if original
                        else model._forward_causal_hazard(sequence, router)
                    )
            except RuntimeError as error:
                outcomes.append(("error", str(error)))
            else:
                outcomes.append(("output", output))
        expected, actual = outcomes
        assert actual[0] == expected[0]
        if lengths == (6, 3):
            assert expected[0] == "error"
        if expected[0] == "error":
            assert actual[1] == expected[1]
        else:
            assert expected[1].dtype == actual[1].dtype == emission_dtype
            assert not expected[1].requires_grad and not actual[1].requires_grad
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            assert torch.isneginf(actual[1]).all()
        assert all(parameter.grad is None for parameter in model.parameters())
