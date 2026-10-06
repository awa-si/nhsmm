from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import duration_log_hazard, filter_model_sequence


def _make_model() -> NHSMM:
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=3,
        causal=True,
        dropout=0.0,
        seed=29,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions(jitter=0.0)
    model.eval()
    return model


def _brute_force_dynamic_hazard_path(model: NHSMM, x: torch.Tensor) -> list[int]:
    sequence = model._build_sequence_set(x)
    T = int(sequence.lengths[0].item())
    K = model.config.n_states
    D = model.config.max_duration

    initial = model.dist.initial.log_matrix(context=sequence.canonical, T=T)[0, 0]
    duration = model.dist.duration.log_matrix(
        context=sequence.contexts,
        T=T,
        soft_dmax=model.duration_logits_bias,
    )[0]
    transition = model.dist.transition.log_matrix(
        context=sequence.contexts,
        T=T,
        soft_dmax=model.duration_logits_bias,
    )[0]
    log_end, log_continue = duration_log_hazard(duration)
    emission = sequence.log_probs[0, :T]

    frontier: list[tuple[list[int], int, int, torch.Tensor]] = [
        ([state], state, 0, initial[state] + emission[0, state]) for state in range(K)
    ]

    for t in range(1, T):
        next_frontier: list[tuple[list[int], int, int, torch.Tensor]] = []
        for path, state, age, score in frontier:
            if age + 1 < D and torch.isfinite(log_continue[t - 1, state, age]):
                next_frontier.append(
                    (
                        path + [state],
                        state,
                        age + 1,
                        score + log_continue[t - 1, state, age] + emission[t, state],
                    )
                )

            if torch.isfinite(log_end[t - 1, state, age]):
                for next_state in range(K):
                    trans = transition[t - 1, state, age, next_state]
                    if torch.isfinite(trans):
                        next_frontier.append(
                            (
                                path + [next_state],
                                next_state,
                                0,
                                score
                                + log_end[t - 1, state, age]
                                + trans
                                + emission[t, next_state],
                            )
                        )
        frontier = next_frontier

    return max(frontier, key=lambda item: float(item[3].detach()))[0]


def test_causal_forward_normalizes_to_filter_trace() -> None:
    torch.manual_seed(29)
    model = _make_model()
    x = torch.randn(2, 6, model.config.n_features)

    sequence = model._build_sequence_set(x)
    alpha = model.forward(sequence)
    trace = filter_model_sequence(model, x)

    log_z = torch.logsumexp(alpha.flatten(2), dim=-1, keepdim=True).unsqueeze(-1)
    normalized = alpha - log_z

    assert torch.allclose(
        normalized,
        trace.log_posterior,
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_viterbi_matches_dynamic_hazard_bruteforce() -> None:
    torch.manual_seed(31)
    model = _make_model()
    x = torch.randn(1, 4, model.config.n_features)

    expected = _brute_force_dynamic_hazard_path(model, x)
    decoded = model.decode(x, first_only=True, verbose=False)

    assert decoded.tolist() == expected


def test_causal_float32_forward_tracks_normalized_filter_over_longer_sequence() -> None:
    torch.manual_seed(53)
    model = _make_model()
    x = torch.randn(2, 17, model.config.n_features)

    sequence = model._build_sequence_set(x)
    alpha = model.forward(sequence)
    trace = filter_model_sequence(model, x)
    log_z = torch.logsumexp(alpha.flatten(2), dim=-1, keepdim=True).unsqueeze(-1)
    normalized = alpha - log_z
    finite = torch.isfinite(normalized) & torch.isfinite(trace.log_posterior)

    torch.testing.assert_close(
        normalized[finite],
        trace.log_posterior[finite],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        normalized.exp(),
        trace.log_posterior.exp(),
        atol=2e-6,
        rtol=2e-6,
    )


def test_causal_forward_keeps_gradient_paths() -> None:
    torch.manual_seed(37)
    model = _make_model()
    x = torch.randn(1, 5, model.config.n_features)

    sequence = model._build_sequence_set(x)
    alpha = model.forward(sequence)
    loss = -torch.logsumexp(alpha[0, -1].flatten(), dim=0)
    loss.backward()

    assert model.dist.duration.logits.grad is not None
    assert model.dist.transition.logits.grad is not None
    assert torch.isfinite(model.dist.duration.logits.grad).all()
    assert torch.isfinite(model.dist.transition.logits.grad).all()


def test_causal_forward_starts_every_sequence_at_age_one() -> None:
    model = _make_model()
    sequence = model._build_sequence_set(torch.randn(2, 5, model.config.n_features))

    alpha = model.forward(sequence)

    assert torch.isfinite(alpha[:, 0, :, 0]).all()
    assert torch.isneginf(alpha[:, 0, :, 1:]).all()
