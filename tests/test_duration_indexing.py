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


def _two_state_model() -> NHSMM:
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=3,
        causal=True,
        transition_type="ergodic",
        seed=11,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions(jitter=0.0)
    model.eval()

    with torch.no_grad():
        model.duration_logits_bias.fill_(20.0)
        model.dist.initial.logits.copy_(torch.tensor([0.0, -40.0]))
        model.dist.duration.logits.fill_(-1000.0)
        # state 0 prefers d=1; state 1 prefers d=2
        model.dist.duration.logits[0, 0] = 0.0
        model.dist.duration.logits[1, 1] = 0.0

        model.dist.transition.logits.fill_(-40.0)
        # From state 0 after d=1, go to state 1.
        model.dist.transition.logits[0, 0, 1] = 40.0
        # From state 0 after d=2, stay in state 0.
        model.dist.transition.logits[0, 1, 0] = 40.0
        # State 1 self-transition for all durations.
        model.dist.transition.logits[1, :, 1] = 40.0

        model.dist.emission.mu.copy_(torch.tensor([[0.0], [10.0]]))
        model.dist.emission.log_var.fill_(-8.0)
    return model


def test_duration_index_zero_means_total_duration_one() -> None:
    model = _deterministic_duration_model(duration=2)
    logp = model.dist.duration.log_matrix(T=1)
    probs = logp.exp()[0, 0, 0]
    assert probs.argmax().item() == 1
    assert probs[1].item() == 1.0


def test_forward_first_segment_score_is_transition_independent() -> None:
    model = _deterministic_duration_model(duration=2)
    # Need two states so transition probabilities are non-trivial.
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=3,
        causal=True,
        transition_type="ergodic",
        seed=13,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions(jitter=0.0)
    model.eval()
    with torch.no_grad():
        model.duration_logits_bias.fill_(20.0)
        model.dist.initial.logits.copy_(torch.tensor([0.0, -40.0]))
        model.dist.duration.logits.fill_(-1000.0)
        model.dist.duration.logits[:, 1] = 0.0
        model.dist.emission.mu.zero_()
        model.dist.emission.log_var.zero_()

    x = torch.zeros(1, 2, 1)
    seq = model._build_sequence_set(x)

    with torch.no_grad():
        model.dist.transition.logits.zero_()
    alpha_a = model.forward(seq)
    score_a = alpha_a[0, 1, 0, 1].clone()

    with torch.no_grad():
        model.dist.transition.logits.copy_(
            torch.tensor(
                [
                    [[40.0, -40.0], [40.0, -40.0], [40.0, -40.0]],
                    [[-40.0, 40.0], [-40.0, 40.0], [-40.0, 40.0]],
                ]
            )
        )
    alpha_b = model.forward(seq)
    score_b = alpha_b[0, 1, 0, 1]

    # A first segment starting at t=0 has no predecessor transition.
    assert torch.allclose(score_a, score_b, atol=1e-6, rtol=1e-6)


def test_viterbi_uses_previous_segment_duration_for_transition() -> None:
    model = _two_state_model()
    # Correct segmentation: state 0 for one bar, then state 1 for two bars.
    x = torch.tensor([[[0.0], [10.0], [10.0]]])
    path = model.decode(x, first_only=True, verbose=False)
    assert path.tolist() == [0, 1, 1]


def test_filter_runtime_share_duration_two_boundaries() -> None:
    model = _deterministic_duration_model(duration=2)
    x = torch.zeros(1, 4, 1)

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


def test_noncausal_viterbi_keeps_predecessor_duration_for_transition(monkeypatch) -> None:
    """A locally worse duration can be globally best after a destination transition."""
    from nhsmm.context import SequenceSet

    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=2,
        causal=False,
        transition_type="ergodic",
        seed=17,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions(jitter=0.0)
    model.eval()

    initial = torch.tensor([[[0.0, -100.0]]])
    duration = torch.tensor(
        [
            [
                [[0.0, -100.0], [-100.0, -100.0]],
                [[0.0, -0.1], [-100.0, -100.0]],
                [[-5.0, -100.0], [0.0, -100.0]],
            ]
        ]
    )
    transition = torch.full((1, 3, 2, 2, 2), -100.0)
    transition[0, 0, 0, 0, 0] = 0.0
    transition[0, 1, 0, 0, 0] = 0.0
    transition[0, 1, 0, 0, 1] = -10.0
    transition[0, 1, 0, 1, 0] = 0.0
    transition[0, 1, 0, 1, 1] = 0.0

    monkeypatch.setattr(model.dist.initial, "log_matrix", lambda **_: initial)
    monkeypatch.setattr(model.dist.duration, "log_matrix", lambda **_: duration)
    monkeypatch.setattr(model.dist.transition, "log_matrix", lambda **_: transition)

    log_probs = torch.zeros(1, 3, 2)
    sequence = SequenceSet(
        sequences=torch.zeros(1, 3, 1),
        lengths=torch.tensor([3]),
        masks=torch.ones(1, 3, 1, dtype=torch.bool),
        contexts=torch.zeros(1, 3, 0),
        canonical=torch.zeros(1, 1, 0),
        log_probs=log_probs,
    )

    # At t=1/state=0, duration 1 has local score 0.0 and duration 2 has -0.1.
    # The boundary to destination state 1 at t=2 scores those durations -10 and
    # 0 respectively. Global MAP must therefore preserve duration 2 and choose
    # state 1 at the final timestep. Collapsing to the locally best duration
    # would incorrectly choose the state-0 alternative with score -5.
    path = model._viterbi(sequence)[0]
    assert path.tolist() == [0, 0, 1]
