from __future__ import annotations

from unittest.mock import patch

import torch

import nhsmm.runtime as runtime_module
from nhsmm import HSMMFilterRuntime, ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from nhsmm.filtering import filter_step, initialize_filter
from tests.reference.hsmm_exact import (
    age_posterior_from_joint,
    causal_prefix_joint_posterior,
    causal_prefix_log_evidence,
    state_posterior_from_joint,
)

# Empirical maxima were measured over the deterministic matrices below before
# fixing these regression bounds. Bounds retain material headroom over the
# observed maxima while remaining well below a probability change of 1e-4.
FLOAT64_LIKELIHOOD_ABS_BOUND = 5e-14
FLOAT64_LOG_POSTERIOR_ABS_BOUND = 5e-13
FLOAT64_PROBABILITY_ABS_BOUND = 5e-14
FLOAT32_LIKELIHOOD_ABS_BOUND = 3e-6
FLOAT32_LOG_POSTERIOR_ABS_BOUND = 1e-5
FLOAT32_PROBABILITY_ABS_BOUND = 2e-6
FLOAT32_LONG_LOG_POSTERIOR_ABS_BOUND = 2e-5
FLOAT32_LONG_PROBABILITY_ABS_BOUND = 1e-5


def _scores(seed: int, *, T: int, K: int, D: int, dtype: torch.dtype, scale: float):
    g = torch.Generator().manual_seed(seed)
    initial = (torch.randn(K, generator=g, dtype=torch.float64) * scale).log_softmax(-1)
    duration = (torch.randn(T, K, D, generator=g, dtype=torch.float64) * scale).log_softmax(-1)
    transition = (torch.randn(T, K, D, K, generator=g, dtype=torch.float64) * scale).log_softmax(-1)
    emission = torch.randn(T, K, generator=g, dtype=torch.float64) * scale * 0.4 - 0.2
    return tuple(x.to(dtype) for x in (initial, duration, transition, emission))


def _sequence(emission: torch.Tensor) -> SequenceSet:
    T, _ = emission.shape
    return SequenceSet(
        sequences=torch.zeros(1, T, 1, dtype=emission.dtype),
        lengths=torch.tensor([T]),
        masks=torch.ones(1, T, 1, dtype=torch.bool),
        contexts=torch.zeros(1, T, 0, dtype=emission.dtype),
        canonical=torch.zeros(1, 1, 0, dtype=emission.dtype),
        log_probs=emission.unsqueeze(0),
    )


def _model(K: int, D: int) -> NHSMM:
    model = NHSMM(
        ModelConfig(
            n_states=K,
            n_features=1,
            max_duration=D,
            causal=True,
            dropout=0.0,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    return model


def _batch_log_posterior(values) -> tuple[torch.Tensor, torch.Tensor]:
    initial, duration, transition, emission = values
    K = initial.shape[0]
    model = _model(K, duration.shape[-1])
    with (
        patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, K)),
        patch.object(model.dist.duration, "log_matrix", return_value=duration.unsqueeze(0)),
        patch.object(model.dist.transition, "log_matrix", return_value=transition.unsqueeze(0)),
    ):
        alpha = model.forward(_sequence(emission))[0]
    evidence = torch.logsumexp(alpha.flatten(1), dim=-1)
    return evidence, alpha - evidence[:, None, None]


def _max_probability_errors(actual_log: torch.Tensor, reference_log: torch.Tensor):
    actual_joint = actual_log.double().exp()
    reference_joint = reference_log.exp()
    return (
        float((actual_joint - reference_joint).abs().max()),
        float((actual_joint.sum(dim=-1) - state_posterior_from_joint(reference_log)).abs().max()),
        float((actual_joint.sum(dim=1) - age_posterior_from_joint(reference_log)).abs().max()),
    )


def test_float64_and_float32_match_exhaustive_reference_within_measured_envelope() -> None:
    maxima = {
        torch.float64: {"ll": 0.0, "log": 0.0, "joint": 0.0, "state": 0.0, "age": 0.0},
        torch.float32: {"ll": 0.0, "log": 0.0, "joint": 0.0, "state": 0.0, "age": 0.0},
    }
    for seed in range(24):
        scale = (0.5, 1.0, 5.0, 10.0)[seed % 4]
        base = _scores(12_000 + seed, T=6, K=2, D=3, dtype=torch.float64, scale=scale)
        ref_log = causal_prefix_joint_posterior(*base)
        ref_evidence = causal_prefix_log_evidence(*base)
        for dtype in (torch.float64, torch.float32):
            values = tuple(x.to(dtype) for x in base)
            evidence, actual_log = _batch_log_posterior(values)
            finite = torch.isfinite(ref_log) & torch.isfinite(actual_log.double())
            m = maxima[dtype]
            m["ll"] = max(m["ll"], float((evidence.double() - ref_evidence).abs().max()))
            if finite.any():
                m["log"] = max(
                    m["log"], float((actual_log.double()[finite] - ref_log[finite]).abs().max())
                )
            joint, state, age = _max_probability_errors(actual_log, ref_log)
            m["joint"] = max(m["joint"], joint)
            m["state"] = max(m["state"], state)
            m["age"] = max(m["age"], age)

    f64 = maxima[torch.float64]
    assert f64["ll"] <= FLOAT64_LIKELIHOOD_ABS_BOUND, f64
    assert f64["log"] <= FLOAT64_LOG_POSTERIOR_ABS_BOUND, f64
    assert max(f64["joint"], f64["state"], f64["age"]) <= FLOAT64_PROBABILITY_ABS_BOUND, f64

    f32 = maxima[torch.float32]
    assert f32["ll"] <= FLOAT32_LIKELIHOOD_ABS_BOUND, f32
    assert f32["log"] <= FLOAT32_LOG_POSTERIOR_ABS_BOUND, f32
    assert max(f32["joint"], f32["state"], f32["age"]) <= FLOAT32_PROBABILITY_ABS_BOUND, f32


def test_float32_long_sequence_batch_filter_and_runtime_remain_within_envelope() -> None:
    maxima = {"log": 0.0, "joint": 0.0, "state": 0.0, "age": 0.0}
    for seed in range(8):
        values = _scores(
            14_000 + seed,
            T=128,
            K=3,
            D=4,
            dtype=torch.float32,
            scale=(1.0, 5.0, 10.0)[seed % 3],
        )
        initial, duration, transition, emission = values
        _, batch = _batch_log_posterior(values)

        state = initialize_filter(initial, emission[0], 4)
        streamed = [state.log_posterior[0]]
        for t in range(1, 128):
            state = filter_step(state, emission[t], duration[t - 1], transition[t - 1])
            streamed.append(state.log_posterior[0])
        streamed = torch.stack(streamed)

        model = _model(3, 4)
        step = {"i": 0}
        with patch.object(model.dist.initial, "log_matrix", return_value=initial.view(1, 1, 3)):

            def fake_emission(_model, observation, context, cache):
                return emission[step["i"]].unsqueeze(0)

            def fake_boundary(_model, context, *, temperature, cache, batch_size=None):
                i = step["i"]
                return duration[i].unsqueeze(0), transition[i].unsqueeze(0)

            with (
                patch.object(runtime_module, "_emission_log_prob", side_effect=fake_emission),
                patch.object(runtime_module, "_boundary_scores", side_effect=fake_boundary),
            ):
                runtime = HSMMFilterRuntime(model)
                runtime_log = []
                for t in range(128):
                    step["i"] = t
                    runtime_log.append(
                        runtime.step(torch.zeros(1, 1), timestamp=t).log_posterior[0]
                    )
                runtime_log = torch.stack(runtime_log)

        for candidate in (streamed, runtime_log):
            finite = torch.isfinite(batch) & torch.isfinite(candidate)
            maxima["log"] = max(
                maxima["log"], float((batch[finite] - candidate[finite]).abs().max())
            )
            bp = batch.exp()
            cp = candidate.exp()
            maxima["joint"] = max(maxima["joint"], float((bp - cp).abs().max()))
            maxima["state"] = max(maxima["state"], float((bp.sum(-1) - cp.sum(-1)).abs().max()))
            maxima["age"] = max(maxima["age"], float((bp.sum(1) - cp.sum(1)).abs().max()))

    assert maxima["log"] <= FLOAT32_LONG_LOG_POSTERIOR_ABS_BOUND, maxima
    assert (
        max(maxima["joint"], maxima["state"], maxima["age"]) <= FLOAT32_LONG_PROBABILITY_ABS_BOUND
    ), maxima


def test_extreme_float32_scores_preserve_dominant_state_and_age_semantics() -> None:
    for seed in range(12):
        base = _scores(15_000 + seed, T=6, K=3, D=3, dtype=torch.float64, scale=12.0)
        reference = causal_prefix_joint_posterior(*base).exp()
        _, actual_log = _batch_log_posterior(tuple(x.float() for x in base))
        actual = actual_log.double().exp()

        ref_state = reference.sum(dim=-1)
        actual_state = actual.sum(dim=-1)
        ref_age = reference.sum(dim=1)
        actual_age = actual.sum(dim=1)

        # Only demand categorical agreement when the Float64 reference has a
        # clear winner; numerically tied semantics are not ordered contracts.
        state_top = ref_state.topk(min(2, ref_state.shape[-1]), dim=-1).values
        if ref_state.shape[-1] == 1:
            state_clear = torch.ones(ref_state.shape[0], dtype=torch.bool)
        else:
            state_clear = (state_top[:, 0] - state_top[:, 1]) > 1e-4
        age_top = ref_age.topk(min(2, ref_age.shape[-1]), dim=-1).values
        age_clear = (age_top[:, 0] - age_top[:, 1]) > 1e-4

        assert torch.equal(actual_state.argmax(-1)[state_clear], ref_state.argmax(-1)[state_clear])
        assert torch.equal(actual_age.argmax(-1)[age_clear], ref_age.argmax(-1)[age_clear])
