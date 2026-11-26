"""Streaming HSMM correctness, mutation-safe caches and independent long prefixes."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch
from nhsmm import NHSMM, HSMMFilterRuntime, ModelConfig
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import _emission_log_prob


def model(dtype=torch.float64, *, tail=False, encoder=False):
    m = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=4,
            causal=True,
            duration_tail=tail,
            allow_self_transitions=False,
            use_context_encoder=encoder,
            context_dim=4 if encoder else None,
            hidden_dim=4,
            dropout=0,
            seed=402,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    m.to(dtype=dtype).eval()
    with torch.no_grad():
        m.duration_logits_bias.zero_()
        m.dist.initial.logits.copy_(torch.tensor([0.3, -0.2, 0.1], dtype=dtype))
        m.dist.duration.logits.copy_(
            torch.tensor(
                [[-0.4, 0.2, 0.7, -0.1], [0.5, -0.3, 0.1, 0.8], [-0.6, 0.6, 0.2, -0.2]], dtype=dtype
            )
        )
        m.dist.transition.logits.copy_(torch.linspace(-0.7, 0.9, 36, dtype=dtype).reshape(3, 4, 3))
        m.dist.emission.mu.copy_(torch.tensor([[-0.8, 0.3], [0.1, -0.5], [0.9, 0.7]], dtype=dtype))
        m.dist.emission.log_var.copy_(
            torch.tensor([[-0.4, 0.2], [0.3, -0.1], [0.1, 0.4]], dtype=dtype)
        )
        for head in (m.dist.initial, m.dist.duration, m.dist.transition):
            head.log_temperature.zero_()
        if tail:
            m.dist.duration.tail_end_probability.copy_(torch.tensor([0.25, 1.0, 0.6], dtype=dtype))
    return m


def array(tensor):
    return tensor.detach().cpu().numpy().astype(np.longdouble)


def softmax(values):
    e = np.exp(values - values.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ingredients(m):
    """Read raw weights only; never use production scoring/hazard/filter routines."""
    pi = softmax(array(m.dist.initial.logits))
    q = softmax(array(m.dist.duration.logits))
    a = array(m.dist.transition.logits).copy()
    for k in range(3):
        a[k, :, k] = -np.inf
    a = softmax(a)
    h = array(m.dist.duration.tail_end_probability) if m.config.duration_tail else None
    matrix = np.zeros((12, 12), dtype=np.longdouble)
    end = np.zeros((3, 4), dtype=np.longdouble)
    for k in range(3):
        for age in range(4):
            hazard = h[k] if h is not None and age == 3 else q[k, age] / q[k, age:].sum()
            end[k, age] = hazard
            for j in range(3):
                matrix[4 * k + age, 4 * j] += hazard * a[k, age, j]
            if age < 3 or h is not None:
                matrix[4 * k + age, 4 * k + min(age + 1, 3)] += 1 - hazard
    return pi, q, h, matrix, end


def reference(m, x):
    pi, q, h, matrix, end = ingredients(m)
    mu = array(m.dist.emission.mu)
    variance = np.maximum(
        np.logaddexp(0, array(m.dist.emission.log_var)), m.dist.emission.min_covar
    )
    raw = array(x)
    log_emission = -0.5 * (
        ((raw[:, None, :] - mu) ** 2 / variance)
        + np.log(variance)
        + np.log(np.longdouble(2) * np.longdouble(str(np.pi)))
    ).sum(-1)
    p = np.zeros((3, 4), dtype=np.longdouble)
    p[:, 0] = pi
    rows = []
    log_z = np.longdouble(0)
    for t in range(len(raw)):
        if t:
            p = (p.reshape(12) @ matrix).reshape(3, 4)
        offset = log_emission[t].max()
        p = p * np.exp(log_emission[t] - offset)[:, None]
        normalizer = p.sum()
        log_z += np.log(normalizer) + offset
        p /= normalizer
        rows.append(p.copy())
    return np.stack(rows).astype(np.float64), float(log_z), (q, h, matrix, end)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tail", [False, True])
def test_1440_row_prefixes_and_reset_against_dense_probability_reference(dtype, tail):
    m = model(dtype, tail=tail)
    t = torch.arange(1440, dtype=dtype)
    x = torch.stack((torch.sin(t / 19) + 0.2 * torch.cos(t / 7), torch.cos(t / 31)), -1)
    expected, evidence, factors = reference(m, x)
    batch = filter_model_sequence(m, x)
    runtime = HSMMFilterRuntime(m)
    streamed = []
    for index, row in enumerate(x):
        streamed.append(runtime.step(row, timestamp=index).log_posterior[0].exp())
        assert runtime.state.observations.shape == (1, 1, 2)
        assert runtime.state.filter_state.log_posterior.shape == (1, 3, 4)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    reference_tensor = torch.tensor(expected, dtype=dtype)
    torch.testing.assert_close(
        batch.log_posterior[0].exp(), reference_tensor, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        torch.stack(streamed), reference_tensor, atol=tolerance, rtol=tolerance
    )
    actual = float(m.log_likelihood(x).detach()[0])
    assert abs(actual - evidence) / len(x) < (1e-5 if dtype == torch.float32 else 1e-10)
    q, h, matrix, end = factors
    posterior = expected[-1].astype(np.longdouble)
    forecast = runtime.forecast_transition()
    expected_next = (posterior.reshape(12) @ matrix).reshape(3, 4).sum(-1)
    torch.testing.assert_close(
        forecast.next_state_prior[0],
        torch.tensor(expected_next.astype(np.float64), dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    assert float(forecast.episode_end_probability[0]) == pytest.approx(
        float((posterior * end).sum()), abs=tolerance, rel=tolerance
    )
    horizons = [1, 4, 25]
    expected_survival = []
    for horizon in horizons:
        total = np.longdouble(0)
        for k in range(3):
            for age in range(4):
                if h is None:
                    conditional = q[k, age + horizon :].sum() / q[k, age:].sum()
                else:
                    exact = min(horizon, 3 - age)
                    conditional = (
                        q[k, age + exact :].sum()
                        / q[k, age:].sum()
                        * (1 - h[k]) ** (horizon - exact)
                    )
                total += posterior[k, age] * conditional
        expected_survival.append(float(total))
    survival = runtime.forecast_survival(horizons)
    torch.testing.assert_close(
        survival.survival_probability[0],
        torch.tensor(expected_survival, dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    runtime.reset()
    assert runtime.state is None
    suffix = x[1312:]
    expected_suffix, _, _ = reference(m, suffix)
    for index, row in enumerate(suffix):
        state = runtime.step(row, timestamp=index)
        torch.testing.assert_close(
            state.log_posterior[0].exp(),
            torch.tensor(expected_suffix[index], dtype=dtype),
            atol=tolerance,
            rtol=tolerance,
        )


class FixedScores(torch.nn.Module):
    """Valid custom scorer with exact zeros, without nonfinite learned parameters."""

    def __init__(self, scores, *, initial=False, tail=None):
        super().__init__()
        self.register_buffer("scores", scores)
        self.register_buffer("tail", tail)
        self.initial = initial
        self.max_duration = 4

    def log_matrix(self, *, T=1, **kwargs):
        return self.scores[None, None].expand(1, 1 if self.initial else T, *self.scores.shape)

    def tail_probability(self):
        return self.tail


@pytest.mark.parametrize("tail", [False, True])
def test_hard_duration_and_transition_support_has_exact_age_cycle(tail):
    m = model(tail=tail)
    initial = torch.full((3,), -torch.inf, dtype=torch.float64)
    initial[0] = 0
    duration = torch.full((3, 4), -torch.inf, dtype=torch.float64)
    duration[:, -1] = 0
    transition = torch.full((3, 4, 3), -torch.inf, dtype=torch.float64)
    for k in range(3):
        transition[k, :, (k + 1) % 3] = 0
    m.dist.initial = FixedScores(initial, initial=True)
    m.dist.duration = FixedScores(
        duration, tail=torch.ones(3, dtype=torch.float64) if tail else None
    )
    m.dist.transition = FixedScores(transition)
    runtime = HSMMFilterRuntime(m)
    for t in range(100):
        state = runtime.step(torch.zeros(2, dtype=torch.float64))
        expected = torch.zeros(3, 4, dtype=torch.float64)
        expected[(t // 4) % 3, t % 4] = 1
        assert torch.equal(state.log_posterior[0].exp(), expected)
    runtime.reset()
    expected = runtime.step(torch.zeros(2, dtype=torch.float64)).log_posterior.exp()
    assert expected[0, 0, 0] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reset", [False, True])
def test_emission_floor_change_invalidates_runtime_cache(dtype, reset):
    m = model(dtype)
    runtime = HSMMFilterRuntime(m)
    runtime.step(torch.zeros(2, dtype=dtype))
    m.dist.emission.min_covar = 4.0
    if reset:
        runtime.reset()
        runtime.step(torch.ones(2, dtype=dtype))
    obs = torch.tensor([[[0.9, -0.6]]], dtype=dtype)
    with torch.inference_mode():
        cached = _emission_log_prob(m, obs, None, runtime._score_cache)
        expected = _emission_log_prob(m, obs, None, None)
    torch.testing.assert_close(cached, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("encoder", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_created_in_inference_mode_can_stream_and_refresh_weights(encoder, dtype):
    with torch.inference_mode():
        m = model(dtype, tail=True, encoder=encoder)
        runtime = HSMMFilterRuntime(m)
        x = torch.tensor([[0.2, -0.3], [0.8, 0.1], [-0.4, 0.7]], dtype=dtype)
        first = runtime.step(x[0])
        assert torch.isfinite(first.state_posterior).all()
        m.duration_logits_bias[0, 0].add_(2)
        m.dist.emission.log_var[0].add_(0.4)
        if encoder:
            m.encoder.encoder.lstm.weight_ih_l0.add_(0.03)
        runtime.reset()
        saved = {key: value.clone() for key, value in m.state_dict().items()}
        actual = torch.stack([runtime.step(row).log_posterior for row in x], 1)
        expected = filter_model_sequence(m, x).log_posterior
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        assert all(torch.equal(value, saved[key]) for key, value in m.state_dict().items())


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), np.datetime64("NaT", "ns")]
)
@pytest.mark.parametrize("started", [False, True])
def test_unordered_or_nonfinite_timestamp_rejected_without_state_advance(bad, started):
    m = model()
    runtime = HSMMFilterRuntime(m)
    if started:
        runtime.step(torch.zeros(2, dtype=torch.float64), timestamp=0)
    before = runtime.state
    with pytest.raises((ValueError, TypeError), match="timestamp"):
        runtime.step(torch.ones(2, dtype=torch.float64), timestamp=bad)
    assert runtime.state is before
    if started:
        good = runtime.step(torch.ones(2, dtype=torch.float64), timestamp=1)
        fresh = HSMMFilterRuntime(m)
        fresh.step(torch.zeros(2, dtype=torch.float64), timestamp=0)
        expected = fresh.step(torch.ones(2, dtype=torch.float64), timestamp=1)
        torch.testing.assert_close(good.log_posterior, expected.log_posterior)


def test_runtime_state_rejects_nan_timestamp():
    runtime = HSMMFilterRuntime(model())
    runtime.step(torch.zeros(2, dtype=torch.float64), timestamp=1)
    with pytest.raises(ValueError, match="timestamp"):
        replace(runtime.state, last_timestamp=float("nan"))


def test_datetime_timestamps_remain_supported_after_rejected_step():
    runtime = HSMMFilterRuntime(model())
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runtime.step(torch.zeros(2, dtype=torch.float64), timestamp=start)
    before = runtime.state
    with pytest.raises(ValueError):
        runtime.step(torch.ones(2, dtype=torch.float64), timestamp=start)
    assert runtime.state is before
    runtime.step(torch.ones(2, dtype=torch.float64), timestamp=start + timedelta(minutes=5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inference_tensor_updates_invalidate_scores_without_reset(dtype):
    from nhsmm.runtime import _boundary_scores

    with torch.inference_mode():
        m = model(dtype, encoder=True)
        runtime = HSMMFilterRuntime(m)
        obs = torch.tensor([[[0.6, -0.2]]], dtype=dtype)
        runtime.step(obs)
        m.duration_logits_bias[0, 0].add_(2)
        m.dist.emission.log_var[0].add_(0.4)
        encoder = m.encoder.encoder
        encoder.lstm.weight_ih_l0.add_(0.03)
        cached = _emission_log_prob(m, obs, None, runtime._score_cache)
        expected = _emission_log_prob(m, obs, None, None)
        torch.testing.assert_close(cached, expected, atol=2e-6, rtol=2e-6)
        cached_duration, _ = _boundary_scores(
            m, None, temperature=None, cache=runtime._score_cache, batch_size=1
        )
        expected_duration, _ = _boundary_scores(m, None, temperature=None, cache=None, batch_size=1)
        torch.testing.assert_close(cached_duration, expected_duration, atol=2e-6, rtol=2e-6)
        state = encoder.initial_stream_state(1, device=obs.device, dtype=dtype)
        actual_context, _ = encoder.stream_step(obs, state)
        expected_context = encoder(obs)
        torch.testing.assert_close(actual_context, expected_context, atol=2e-6, rtol=2e-6)
