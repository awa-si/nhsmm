from __future__ import annotations

import torch

from nhsmm.config import ModelConfig
from nhsmm.models.base import NHSMM
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import HSMMFilterRuntime
from nhsmm.survival import active_episode_survival_forecast


def _deterministic_duration_model(duration: int = 2) -> NHSMM:
    cfg = ModelConfig(
        n_states=1,
        n_features=1,
        max_duration=3,
        causal=True,
        transition_type="ergodic",
        seed=7,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions(jitter=0.0)
    model.eval()

    with torch.no_grad():
        model.dist.initial.logits.zero_()
        model.dist.duration.logits.fill_(-1000.0)
        model.dist.duration.logits[0, duration - 1] = 0.0
        model.dist.transition.logits.zero_()
        model.dist.emission.mu.zero_()
        model.dist.emission.log_var.zero_()
        model.duration_logits_bias.fill_(20.0)
    return model


def test_duration_index_zero_means_total_duration_one() -> None:
    model = _deterministic_duration_model(duration=2)
    logp = model.dist.duration.log_matrix(T=1)
    probs = logp.exp()[0, 0, 0]
    assert probs.argmax().item() == 1
    assert probs[1].item() == 1.0


def test_forward_accepts_first_segment_of_duration_two() -> None:
    model = _deterministic_duration_model(duration=2)
    x = torch.zeros(1, 2, 1)
    seq = model._build_sequence_set(x)
    alpha = model.forward(seq)

    # Endpoint t=1 with total duration 2 must be a valid first segment.
    assert torch.isfinite(alpha[0, 1, 0, 1])


def test_viterbi_and_filter_runtime_share_duration_two_boundaries() -> None:
    model = _deterministic_duration_model(duration=2)
    x = torch.zeros(1, 4, 1)

    path = model.decode(x, first_only=True, verbose=False)
    assert path.shape == (4,)

    trace = filter_model_sequence(model, x)
    ages = trace.log_posterior[0].exp().sum(dim=1).argmax(dim=-1) + 1
    assert ages.tolist() == [1, 2, 1, 2]

    runtime = HSMMFilterRuntime(model)
    runtime_ages = []
    for t in range(4):
        state = runtime.step(x[:, t])
        runtime_ages.append(int(state.age_posterior.argmax(dim=-1).item()) + 1)
    assert runtime_ages == [1, 2, 1, 2]


def test_survival_horizon_one_matches_duration_two_age_contract() -> None:
    model = _deterministic_duration_model(duration=2)
    runtime = HSMMFilterRuntime(model)
    x0 = torch.zeros(1, 1)
    x1 = torch.zeros(1, 1)

    state0 = runtime.step(x0)
    f0 = active_episode_survival_forecast(state0, runtime.state.duration_log_prob, [1])
    assert torch.allclose(f0.survival_probability, torch.ones_like(f0.survival_probability))

    state1 = runtime.step(x1)
    f1 = active_episode_survival_forecast(state1, runtime.state.duration_log_prob, [1])
    assert torch.allclose(f1.survival_probability, torch.zeros_like(f1.survival_probability))
