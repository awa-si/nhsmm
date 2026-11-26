"""Same-parameter serial/parallel HMM score and gradient envelopes."""

import pytest
import torch
from nhsmm.filtering import filter_model_sequence
from nhsmm.models.forward import _hmm_prefix_log_posteriors

from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture


@pytest.mark.parametrize("length", [1, 2, 3, 5, 17])
def test_matrix_scan_matches_scalar_prefix_probability_products_and_has_no_lookahead(length):
    torch.manual_seed(length)
    pi = torch.randn(3, dtype=torch.float64).log_softmax(-1)
    e = torch.randn(length, 3, dtype=torch.float64)
    a = torch.randn(max(0, length - 1), 3, 3, dtype=torch.float64).log_softmax(-1)
    actual = _hmm_prefix_log_posteriors(pi, e, a)
    current = (pi + e[0]).softmax(-1)
    expected = [current]
    for t in range(1, length):
        current = (current @ a[t - 1].exp()) * e[t].exp()
        current = current / current.sum()
        expected.append(current)
    torch.testing.assert_close(actual.exp(), torch.stack(expected), atol=1e-12, rtol=1e-10)
    if length > 2:
        altered = e.clone()
        altered[2:] += torch.tensor([100.0, -200.0, 70.0])
        torch.testing.assert_close(
            _hmm_prefix_log_posteriors(pi, altered, a)[:2], actual[:2], atol=1e-12, rtol=1e-10
        )


def test_scan_and_serial_raw_parameter_gradients_are_equal(monkeypatch):
    import nhsmm.models.forward as mod

    model, x, c = fixture(1, False)
    z = torch.tensor([0, 1, 2, 0])
    parameters = [p for p in model.parameters() if p.requires_grad]
    fast = model.supervised_filter_log_score(x, z, context=c).sum()
    left = torch.autograd.grad(fast, parameters, allow_unused=True)
    monkeypatch.setattr(mod, "_SMALL_HMM_SCAN_MAX_STATES", 0)
    slow = model.supervised_filter_log_score(x, z, context=c).sum()
    right = torch.autograd.grad(slow, parameters, allow_unused=True)
    torch.testing.assert_close(fast, slow, atol=1e-12, rtol=1e-10)
    for p, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_long_float32_scan_matches_native_filter_within_existing_envelope():
    model, _, _ = fixture(1, False)
    model.float().eval()
    torch.manual_seed(113)
    x = torch.randn(1440, 2)
    data = model._build_sequence_set(x)
    pi = model.dist.initial.log_matrix(T=1)[0, 0]
    a = model.dist.transition.log_matrix(T=len(x))[0, :-1, :, 0, :]
    fast = _hmm_prefix_log_posteriors(pi, data.log_probs[0], a)
    native = filter_model_sequence(model, x).state_posterior[0]
    torch.testing.assert_close(fast.exp(), native, atol=1e-5, rtol=0)


def test_hard_support_scan_keeps_impossible_states_and_finite_reachable_gradients():
    pi = torch.tensor([0.0, -torch.inf], dtype=torch.float64)
    e = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[[0.0, -torch.inf], [-torch.inf, 0.0]]] * 2, dtype=torch.float64)
    p = _hmm_prefix_log_posteriors(pi, e, a)
    torch.testing.assert_close(p.exp(), torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64))
    g = torch.autograd.grad(p[:, 0].sum(), e)[0]
    assert torch.isfinite(g).all()
