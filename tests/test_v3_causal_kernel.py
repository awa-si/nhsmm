"""Independent path-mass checks for the shared causal frontier and evidence."""

from __future__ import annotations

import pytest
import torch

from nhsmm.filtering import (
    _causal_filter_step,
    _duration_log_hazard_from_log_p,
    _initialize_causal_frontier,
    _normalize_causal_frontier,
)


def _enumerated_prefixes(initial, duration, transition, emission, tail):
    """Enumerate distinct continuation/renewal paths in probability space."""
    T, K, D = duration.shape
    paths = [(k, 0, initial[k] * emission[0, k]) for k in range(K)]
    results = []
    for t in range(T):
        total = torch.stack([p for _, _, p in paths]).sum()
        rows = []
        for k in range(K):
            cells = []
            for a in range(D):
                masses = [p for state, age, p in paths if (state, age) == (k, a)]
                cells.append(torch.stack(masses).sum() if masses else total.new_zeros(()))
            rows.append(torch.stack(cells))
        results.append((torch.stack(rows) / total, total.log()))
        if t + 1 == T:
            break
        advanced = []
        for k, a, mass in paths:
            survival = duration[t, k, a:].sum()
            if float(survival.detach()) == 0:
                continue
            if tail is not None and a == D - 1:
                end = tail[k]
                continuation = 1 - end
            else:
                end = duration[t, k, a] / survival
                continuation = duration[t, k, a + 1 :].sum() / survival
            if float(continuation.detach()) > 0:
                advanced.append((k, min(a + 1, D - 1), mass * continuation * emission[t + 1, k]))
            if float(end.detach()) > 0:
                row = transition[t, k, a if transition.shape[2] > 1 else 0]
                for destination in range(K):
                    value = mass * end * row[destination] * emission[t + 1, destination]
                    if float(value.detach()) > 0:
                        advanced.append((destination, 0, value))
        paths = advanced
    return results


@pytest.mark.parametrize("D", [1, 3])
@pytest.mark.parametrize("tailed", [False, True])
@pytest.mark.parametrize("age_dependent", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_shared_frontier_evidence_and_gradients_match_enumerated_paths(
    D, tailed, age_dependent, masked
):
    T, K = 4, 2
    generator = torch.Generator().manual_seed(77 + D)
    initial = torch.randn(K, dtype=torch.float64, generator=generator).requires_grad_()
    duration = torch.randn(T, K, D, dtype=torch.float64, generator=generator)
    if masked and D > 1:
        duration[..., 0] = -torch.inf
    duration.requires_grad_()
    transition = torch.randn(
        T, K, D if age_dependent else 1, K, dtype=torch.float64, generator=generator
    ).requires_grad_()
    emission = (torch.randn(T, K, dtype=torch.float64, generator=generator) - 1).requires_grad_()
    tail_raw = torch.tensor([-1.2, 0.7], dtype=torch.float64, requires_grad=True)
    tail = tail_raw.sigmoid() if tailed else None
    lp, ld, lt = initial.log_softmax(-1), duration.log_softmax(-1), transition.log_softmax(-1)
    expected = _enumerated_prefixes(lp.exp(), ld.exp(), lt.exp(), emission.exp(), tail)
    log_end, log_continue = _duration_log_hazard_from_log_p(ld, tail)
    frontier, cumulative = _initialize_causal_frontier(lp[None], emission[0:1], D)
    for t in range(T):
        if t:
            frontier, increment = _causal_filter_step(
                frontier,
                emission[t : t + 1],
                log_end[t - 1 : t],
                log_continue[t - 1 : t],
                lt[t - 1 : t],
                tail_enabled=tailed,
            )
            cumulative = cumulative + increment
        torch.testing.assert_close(frontier.exp()[0], expected[t][0], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(cumulative[0], expected[t][1], atol=2e-12, rtol=2e-12)
        assert torch.equal(torch.isneginf(frontier[0]), expected[t][0] == 0)
    leaves = (initial, duration, transition, emission) + ((tail_raw,) if tailed else ())
    actual_grad = torch.autograd.grad(
        cumulative.sum(), leaves, allow_unused=True, retain_graph=True
    )
    reference_grad = torch.autograd.grad(expected[-1][1], leaves, allow_unused=True)
    for leaf, actual, reference in zip(leaves, actual_grad, reference_grad):
        actual = torch.zeros_like(leaf) if actual is None else actual
        reference = torch.zeros_like(leaf) if reference is None else reference
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frontier_retains_mass_under_large_finite_offsets(dtype):
    scores = torch.tensor([[[-100000.5, -torch.inf], [-100002.5, -100004.0]]], dtype=dtype)
    posterior, evidence = _normalize_causal_frontier(scores)
    torch.testing.assert_close(posterior.flatten(1).logsumexp(1), torch.zeros(1, dtype=dtype))
    torch.testing.assert_close(evidence, scores.flatten(1).logsumexp(1))
    assert torch.isneginf(posterior[0, 0, 1])


@pytest.mark.parametrize("value", [-torch.inf, torch.inf, torch.nan])
def test_no_posterior_is_returned_without_finite_evidence(value):
    with pytest.raises(ValueError, match="evidence.*probability mass"):
        _normalize_causal_frontier(torch.full((1, 2, 3), value))


def test_empty_active_batch_has_no_evidence_or_posterior_rows():
    initial = torch.empty(0, 2)
    posterior, evidence = _initialize_causal_frontier(initial, initial, 3)
    assert posterior.shape == (0, 2, 3)
    assert evidence.shape == (0,)
    result, increment = _causal_filter_step(
        posterior,
        initial,
        torch.empty(0, 2, 3),
        torch.empty(0, 2, 3),
        torch.empty(0, 2, 3, 2),
        tail_enabled=True,
    )
    assert result.shape == posterior.shape and increment.shape == evidence.shape
