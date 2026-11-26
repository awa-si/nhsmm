"""HMM specializations against unchanged general recurrences and raw paths."""

import pytest
import torch
from nhsmm import NHSMM, HSMMFilterRuntime, ModelConfig
from nhsmm import filtering as f

from tests.ml_stack.nhsmm.reference.neural_parameter_reference import enumerate_weights
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


@pytest.mark.parametrize("k", [2, 3, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_one_step_inference_matches_general_kernel(k, dtype, monkeypatch):
    torch.manual_seed(19)
    previous = f.initialize_filter(
        torch.randn(2, k, dtype=dtype), torch.randn(2, k, dtype=dtype), 1
    )
    emission = torch.randn(2, k, dtype=dtype)
    duration = torch.zeros(2, k, 1, dtype=dtype)
    transition = torch.randn(2, k, 1, k, dtype=dtype).log_softmax(-1)
    end, continuation = f._duration_log_hazard_from_log_p(duration)
    expected, _ = f._causal_filter_step(
        previous.log_posterior, emission, end, continuation, transition, tail_enabled=False
    )
    original = f._duration_log_hazard_from_log_p
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(f, "_duration_log_hazard_from_log_p", counted)
    with torch.inference_mode():
        actual = f._filter_step_normalized(previous, emission, duration, transition)
    assert not calls
    torch.testing.assert_close(actual.log_posterior, expected, rtol=0, atol=0)
    with torch.enable_grad():
        f._filter_step_normalized(previous, emission, duration, transition)
    assert len(calls) == 1


@pytest.mark.parametrize("first_available", [False, True])
def test_three_state_neural_hmm_batch_matches_independent_raw_paths(first_available):
    model, x, context = fixture(1, False, first_available)
    raw = parts(model, x, context)
    actual = f.filter_model_sequence(model, x, context=context).state_posterior[0]
    for length in range(1, len(x) + 1):
        paths = enumerate_weights(raw, hmm=True, length=length)
        denominator = torch.stack([w for _, w, _ in paths]).sum()
        expected = torch.stack(
            [sum(w for p, w, _ in paths if p[-1] == state) / denominator for state in range(3)]
        )
        torch.testing.assert_close(actual[length - 1], expected, rtol=0, atol=2e-12)


def test_invalid_reachable_duration_support_still_rejected():
    previous = f.initialize_filter(torch.zeros(2), torch.zeros(2), 1)
    with torch.inference_mode(), pytest.raises(ValueError, match="zero survival"):
        f._filter_step_normalized(
            previous,
            torch.zeros(1, 2),
            torch.tensor([[[-torch.inf], [0.0]]]),
            torch.zeros(1, 2, 1, 2).log_softmax(-1),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_hmm_scan_preserves_ragged_empty_and_causal_prefixes(dtype):
    model = NHSMM(ModelConfig.for_mode("hmm", n_states=3, n_features=2, seed=19), device="cpu")
    model.initialize_distributions()
    model.to(dtype=dtype).eval()
    torch.manual_seed(31)
    xs = [torch.randn(n, 2, dtype=dtype) for n in [0, 1, 19]]
    trace = f.filter_model_sequence(model, xs)
    assert trace.lengths.tolist() == [0, 1, 19]
    atol = 2e-5 if dtype == torch.float32 else 2e-12
    for b, x in enumerate(xs):
        rt = HSMMFilterRuntime(model)
        for t, row in enumerate(x):
            torch.testing.assert_close(
                rt.step(row).state_posterior[0], trace.state_posterior[b, t], rtol=0, atol=atol
            )
        assert torch.equal(
            trace.state_posterior[b, len(x) :], torch.zeros_like(trace.state_posterior[b, len(x) :])
        )
    changed = xs[-1].clone()
    changed[8:] += 100
    altered = f.filter_model_sequence(model, changed).state_posterior[0]
    torch.testing.assert_close(altered[:8], trace.state_posterior[-1, :8], rtol=0, atol=0)
