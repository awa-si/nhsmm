"""Independent long-horizon and supervised-derivative gates for the HMM scan."""

import numpy as np
import pytest
import torch
from nhsmm.models.forward import _hmm_prefix_log_posteriors


def _probability_reference(initial, emission, transition):
    """Long-double sequential Bayes chain, independent of production log helpers."""
    pi, ell, edges = [
        value.detach().cpu().numpy().astype(np.longdouble)
        for value in (initial, emission, transition)
    ]
    likelihood = np.exp(ell - ell.max(axis=-1, keepdims=True))
    matrix = np.exp(edges)
    posterior = np.exp(pi)
    rows = []
    for t, density in enumerate(likelihood):
        if t:
            posterior = posterior @ matrix[t - 1]
        posterior = posterior * density
        assert posterior.sum() > 0
        posterior /= posterior.sum()
        rows.append(posterior.copy())
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("scenario", ["dense", "persistent", "hard_support", "common_offset"])
def test_8193_hmm_prefixes_against_long_double_chain(dtype, k, scenario, record_property):
    generator = torch.Generator().manual_seed(901 + k)
    length = 8193  # Crosses a power-of-two scan boundary.
    initial = torch.randn(k, generator=generator, dtype=dtype).log_softmax(-1)
    emission = torch.randn(length, k, generator=generator, dtype=dtype) * 0.3
    logits = torch.randn(length - 1, k, k, generator=generator, dtype=dtype)
    if scenario == "persistent":
        logits = logits * 0.1 + torch.eye(k, dtype=dtype) * 12
    elif scenario == "hard_support":
        allowed = torch.eye(k, dtype=torch.bool) | torch.eye(k, dtype=torch.bool).roll(1, -1)
        logits = logits.masked_fill(~allowed, -torch.inf)
        emission[::17, 0] = -torch.inf
    elif scenario == "common_offset":
        emission -= 1_000_000
    transition = logits.log_softmax(-1)
    expected = _probability_reference(initial, emission, transition)
    with torch.inference_mode():
        log_actual = _hmm_prefix_log_posteriors(initial, emission, transition)
    actual = log_actual.exp().cpu().numpy()
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    np.testing.assert_allclose(actual.sum(-1), 1, rtol=0, atol=tolerance)
    if scenario == "hard_support":
        assert np.all(actual[::17, 0] == 0)
    record_property("maximum_posterior_error", float(np.max(np.abs(actual - expected))))


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("hard_support", [False, True])
def test_supervised_scan_gradient_and_curvature_against_probability_chain(k, hard_support):
    generator = torch.Generator().manual_seed(392)
    length = 17
    raw = [
        torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in [(k,), (length, k), (length - 1, k, k)]
    ]
    pi, emission, logits = raw
    if hard_support:
        allowed = torch.eye(k, dtype=torch.bool) | torch.eye(k, dtype=torch.bool).roll(1, -1)
        logits = logits.masked_fill(~allowed, -torch.inf)
    transition = logits.log_softmax(-1)
    log_initial = pi.log_softmax(-1)
    scanned = _hmm_prefix_log_posteriors(log_initial, emission, transition)
    # Independent differentiable probability-space recurrence; no shared scan/log helpers.
    reference_raw = [v.detach().clone().requires_grad_() for v in raw]
    ref_pi, ref_emission, ref_logits = reference_raw
    if hard_support:
        ref_logits = ref_logits.masked_fill(~allowed, -torch.inf)
    weights = ref_emission.exp()
    matrices = ref_logits.exp()
    matrices = matrices / matrices.sum(-1, keepdim=True)
    posterior = ref_pi.exp()
    posterior = posterior / posterior.sum()
    rows = []
    for t in range(length):
        if t:
            posterior = posterior @ matrices[t - 1]
        posterior = posterior * weights[t]
        posterior = posterior / posterior.sum()
        rows.append(posterior)
    reference = torch.stack(rows).log()
    labels = torch.arange(length) % k
    indices = torch.arange(length)
    actual_loss = -scanned[indices, labels].mean()
    expected_loss = -reference[indices, labels].mean()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=2e-12)
    actual_grad = torch.autograd.grad(actual_loss, raw, create_graph=True)
    expected_grad = torch.autograd.grad(expected_loss, reference_raw, create_graph=True)
    directions = [torch.randn(v.shape, generator=generator, dtype=v.dtype) for v in raw]
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    actual_hvp = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(actual_grad, directions, strict=True)), raw
    )
    expected_hvp = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(expected_grad, directions, strict=True)), reference_raw
    )
    for actual, expected in zip(actual_hvp, expected_hvp, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
