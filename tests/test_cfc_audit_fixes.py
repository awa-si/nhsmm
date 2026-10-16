from __future__ import annotations

import math

import pytest
import torch

from nhsmm import HSMMFilterRuntime, ModelConfig, NHSMM
from nhsmm.convergence import Convergence
from nhsmm.filtering import (
    HSMMFilterState,
    HSMMFilterTrace,
    _filter_step_normalized,
    filter_step,
    initialize_filter,
    next_episode_end_probability,
)


@pytest.mark.parametrize("kernel", [filter_step, _filter_step_normalized])
@pytest.mark.parametrize("mixed", [False, True])
def test_impossible_evidence_raises_before_state_normalization(kernel, mixed):
    batch = 2 if mixed else 1
    state = initialize_filter(torch.zeros(batch, 1), torch.zeros(batch, 1), 1)
    original = state.log_posterior.clone()
    emission = torch.zeros(batch, 1)
    emission[-1] = -math.inf
    with pytest.raises(ValueError, match="probability mass"):
        kernel(state, emission, torch.zeros(batch, 1, 1), torch.zeros(batch, 1, 1, 1))
    torch.testing.assert_close(state.log_posterior, original, atol=0, rtol=0)


def test_disjoint_support_cannot_create_a_posterior():
    state = initialize_filter(torch.tensor([[0.0, -math.inf]]), torch.zeros(1, 2), 1)
    transition = torch.eye(2).log().reshape(1, 2, 1, 2)
    with pytest.raises(ValueError, match="probability mass"):
        filter_step(state, torch.tensor([[-math.inf, 0.0]]), torch.zeros(1, 2, 1), transition)


def test_runtime_retains_last_accepted_state_when_kernel_rejects_evidence(monkeypatch):
    import nhsmm.runtime as runtime_module

    model = NHSMM(ModelConfig(n_states=1, n_features=1, max_duration=1, causal=True))
    model.initialize_distributions()
    model.eval()
    runtime = HSMMFilterRuntime(model)
    runtime.step(torch.zeros(1))
    previous = runtime.state
    # Inject an impossible scorer result to exercise the shared kernel guard.
    # The built-in Gaussian scorer already rejects its own non-finite outputs.
    monkeypatch.setattr(
        runtime_module, "_emission_log_prob", lambda *args, **kwargs: torch.full((1, 1), -math.inf)
    )
    with pytest.raises(ValueError, match="probability mass"):
        runtime.step(torch.ones(1))
    assert runtime.state is previous


@pytest.mark.parametrize("mass", [0.5, 2.0])
def test_public_state_rejects_unnormalized_mass(mass):
    with pytest.raises(ValueError, match="normalized"):
        HSMMFilterState(torch.full((1, 1, 1), math.log(mass)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_state_accepts_normalized_mass_and_forecast_is_a_probability(dtype):
    values = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
    values = torch.log_softmax(values.flatten(1), dim=1).reshape_as(values)
    state = HSMMFilterState(values)
    probability = next_episode_end_probability(state, torch.zeros(2, 3, 4, dtype=dtype))
    assert ((probability >= 0) & (probability <= 1)).all()
    torch.testing.assert_close(state.state_posterior.sum(-1), torch.ones(2, dtype=dtype))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, math.log(2)])
def test_trace_rejects_invalid_valid_rows(value):
    trace = torch.full((1, 2, 1, 1), -math.inf)
    trace[:, 0] = value
    with pytest.raises(ValueError):
        HSMMFilterTrace(trace, torch.tensor([1]))


@pytest.mark.parametrize("value", [0.0, math.nan, math.inf])
def test_trace_requires_negative_infinity_padding(value):
    trace = torch.zeros(1, 2, 1, 1)
    trace[:, 1] = value
    with pytest.raises(ValueError, match="padding"):
        HSMMFilterTrace(trace, torch.tensor([1]))


@pytest.mark.parametrize(
    "shape,lengths",
    [
        ((2, 0, 1, 1), [0, 0]),
        ((0, 3, 1, 1), []),
        ((2, 3, 1, 1), [0, 1]),
    ],
)
def test_trace_preserves_empty_and_padded_sequences(shape, lengths):
    trace = torch.full(shape, -math.inf)
    if shape[0] and shape[1]:
        trace[1, 0] = 0.0
    state = HSMMFilterTrace(trace, torch.tensor(lengths, dtype=torch.long))
    assert not torch.isnan(state.state_posterior).any()


@pytest.mark.parametrize("shape", [(1, 2, 0, 1), (1, 2, 1, 0)])
def test_trace_rejects_empty_latent_dimensions(shape):
    with pytest.raises(ValueError):
        HSMMFilterTrace(torch.empty(shape), torch.tensor([0]))


def test_plateau_preserves_small_improvements_on_a_large_score():
    monitor = Convergence(1, 4, mode="plateau", plateau_window=3, plateau_tol=0, verbose=False)
    scores = [100000000.0, 100000001.0, 100000002.0]
    flags = [monitor.update(score, i, 0) for i, score in enumerate(scores)]
    assert not any(flags)
    assert monitor.scores[:3, 0].tolist() == scores
    assert int(monitor.best_iters[0]) == 2


def test_delta_recognizes_identical_python_float_scores():
    monitor = Convergence(1, 4, patience=2, tol=0, rel_tol=0, verbose=False)
    flags = [monitor.update(100000000.1, i, 0) for i in range(3)]
    assert flags == [False, False, True]
    assert monitor.deltas[1:3, 0].tolist() == [0.0, 0.0]


def test_scores_and_best_score_preserve_fractional_input_precision():
    monitor = Convergence(1, 4, verbose=False)
    monitor.update(100000000.1, 0, 0)
    assert float(monitor.scores[0, 0]) == 100000000.1
    assert float(monitor.best_scores[0]) == 100000000.1


@pytest.mark.parametrize("seed", [-(2**63) - 1, 2**64])
def test_model_config_rejects_seeds_outside_torch_domain(seed):
    with pytest.raises(ValueError, match="seed"):
        ModelConfig(n_states=1, n_features=1, seed=seed)


@pytest.mark.parametrize("seed", [-(2**63), -1, 0, 2**64 - 1])
def test_model_seed_domain_matches_torch_including_negative_seeds(seed):
    config = ModelConfig(n_states=1, n_features=1, seed=seed)
    model = NHSMM(config, device="cpu")
    assert model.config.seed == seed


@pytest.mark.parametrize("offset", [-1e4, -1e7])
@pytest.mark.parametrize("advance", [False, True])
def test_large_finite_emission_offsets_preserve_normalized_mass(offset, advance):
    emission = torch.tensor([[offset - 0.25, offset - 2.5]])
    if advance:
        previous = initialize_filter(torch.zeros(1, 2), torch.zeros(1, 2), 3)
        state = filter_step(previous, emission, torch.zeros(1, 2, 3), torch.zeros(1, 2, 3, 2))
    else:
        state = initialize_filter(torch.zeros(1, 2), emission, 3)
    torch.testing.assert_close(
        state.log_posterior.flatten(1).logsumexp(1), torch.zeros(1), atol=1e-6, rtol=0
    )
