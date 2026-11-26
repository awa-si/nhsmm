"""Viterbi decisions must retain small path differences beside common density offsets."""

import math

import pytest
import torch
from nhsmm import NHSMM, ModelConfig

from tests.ml_stack.nhsmm.reference.neural_parameter_reference import enumerate_weights
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


def identical_emissions(dtype, causal, duration=1, tail=False):
    model = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=duration,
            duration_tail=tail,
            causal=causal,
            dropout=0,
            seed=13,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.to(dtype=dtype).eval()
    with torch.no_grad():
        model.dist.initial.logits.copy_(torch.tensor([0.1, 0.9], dtype=dtype).log())
        a = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=dtype)
        model.dist.transition.logits.copy_(a[:, None].expand(2, duration, 2).log())
        model.dist.duration.logits.zero_()
        model.duration_logits_bias.zero_()
        model.dist.emission.mu.zero_()
        model.dist.emission.log_var.fill_(math.log(math.expm1(1.0)))
        for head in (model.dist.initial, model.dist.transition, model.dist.duration):
            head.log_temperature.zero_()
        if tail:
            model.dist.duration.tail_end_probability.fill_(0.25)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("duration,tail", [(1, False), (3, False), (1, True), (3, True)])
def test_common_gaussian_density_does_not_erase_map_prior(dtype, causal, duration, tail):
    model = identical_emissions(dtype, causal, duration, tail)
    large = 1e4 if dtype == torch.float32 else 1e9
    # Every segmentation can replace all state labels by 1 without reducing its
    # transition probability; pi[1] > pi[0]. Thus the unique MAP labels are all 1.
    for value in (0.0, large, -large):
        xs = [torch.full((n, 1), value, dtype=dtype) for n in (0, 1, 4)]
        paths = model.predict(xs, verbose=False)
        assert [p.tolist() for p in paths] == [[], [1], [1, 1, 1, 1]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duration,tail", [(2, False), (1, True), (3, True)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_dynamic_hsmm_offset_invariance_matches_raw_weight_path_max(dtype, duration, tail, sign):
    model, x, context = fixture(duration, tail, first_available=False, allow_self=False)
    model.to(dtype=dtype)
    # Float64 raw-weight reference is evaluated independently of decoder scores.
    raw = parts(
        model,
        x.to(dtype),
        type(context)(context.emission.to(dtype), context.boundary.to(dtype), context.available),
    )
    emission = torch.tensor([[0, -8, -16], [-8, 0, -8], [0, 0, -8], [-16, -8, 0]], dtype=dtype)
    reference = list(raw)
    reference[1] = emission.double().exp()
    reference = tuple(v.double() if v is not None else None for v in reference)
    rows = enumerate_weights(reference, hmm=False, length=4)
    optimum = max(float(w.detach()) for _, w, _ in rows)
    assert optimum > 0
    c = type(context)(context.emission.to(dtype), context.boundary.to(dtype), context.available)
    data = model._build_sequence_set(x.to(dtype), context=c)
    original = data.log_probs.clone()
    offset = sign * (2**24 if dtype == torch.float32 else 2**52)
    for common in (0, offset):
        data.log_probs = (emission + common).unsqueeze(0)
        predicted = tuple(model._viterbi(data)[0].tolist())
        score = max(float(w.detach()) for p, w, _ in rows if p == predicted)
        assert score == pytest.approx(optimum, abs=1e-12, rel=1e-10)
    # Caller-owned sequence tensors must not be normalized in place.
    assert torch.equal(data.log_probs, (emission + offset).unsqueeze(0))
    assert original.shape == data.log_probs.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_long_causal_frontier_retains_weak_state_preference(dtype):
    model = identical_emissions(dtype, True)
    delta = 1e-5 if dtype == torch.float32 else 1e-14
    with torch.no_grad():
        # Every destination is independent of the previous state. The unique
        # per-row optimum is state 1 even when its margin is small.
        row = torch.tensor([0.5 - delta, 0.5 + delta], dtype=dtype)
        model.dist.transition.logits.copy_(row.log().view(1, 1, 2).expand(2, 1, 2))
    actual = model.predict(torch.zeros(1440, 1, dtype=dtype), verbose=False)[0]
    assert torch.equal(actual, torch.ones(1440, dtype=torch.long))
