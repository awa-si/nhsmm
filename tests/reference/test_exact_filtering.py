from __future__ import annotations

import torch

from tests.reference.hsmm_exact import causal_prefix_joint_posterior


def test_exact_filtering_deterministic_duration_two_age_cycle() -> None:
    initial = torch.zeros(1, dtype=torch.float64)
    duration = torch.full((4, 1, 2), float("-inf"), dtype=torch.float64)
    duration[..., 1] = 0.0
    transition = torch.zeros(4, 1, 2, 1, dtype=torch.float64)
    emission = torch.zeros(4, 1, dtype=torch.float64)

    posterior = causal_prefix_joint_posterior(initial, duration, transition, emission).exp()
    ages = posterior.sum(dim=1).argmax(dim=-1) + 1
    assert ages.tolist() == [1, 2, 1, 2]
    torch.testing.assert_close(posterior.sum(dim=(1, 2)), torch.ones(4, dtype=torch.float64))


def test_exact_filtering_self_transition_resets_age() -> None:
    initial = torch.tensor([0.0, float("-inf")], dtype=torch.float64)
    duration = torch.full((3, 2, 2), float("-inf"), dtype=torch.float64)
    duration[..., 0] = 0.0  # every episode has total duration one
    transition = torch.full((3, 2, 2, 2), float("-inf"), dtype=torch.float64)
    transition[:, 0, :, 0] = 0.0  # state 0 always starts a new state-0 episode
    transition[:, 1, :, 1] = 0.0
    emission = torch.zeros(3, 2, dtype=torch.float64)
    emission[:, 1] = float("-inf")

    posterior = causal_prefix_joint_posterior(initial, duration, transition, emission).exp()
    assert posterior[:, 0, 0].tolist() == [1.0, 1.0, 1.0]
    assert posterior[:, 0, 1].tolist() == [0.0, 0.0, 0.0]


def test_exact_filtering_impossible_duration_support_is_exact() -> None:
    initial = torch.zeros(1, dtype=torch.float64)
    duration = torch.full((2, 1, 2), float("-inf"), dtype=torch.float64)
    duration[..., 1] = 0.0  # duration one impossible; duration two certain
    transition = torch.zeros(2, 1, 2, 1, dtype=torch.float64)
    emission = torch.zeros(2, 1, dtype=torch.float64)

    posterior = causal_prefix_joint_posterior(initial, duration, transition, emission)
    assert posterior[0, 0, 0].item() == 0.0
    assert torch.isneginf(posterior[0, 0, 1])
    assert torch.isneginf(posterior[1, 0, 0])
    assert posterior[1, 0, 1].item() == 0.0


def test_exact_prefix_evidence_and_episode_end_probability() -> None:
    from tests.reference.hsmm_exact import (
        causal_prefix_episode_end_probability,
        causal_prefix_log_evidence,
    )

    initial = torch.zeros(1, dtype=torch.float64)
    duration = torch.full((4, 1, 2), float("-inf"), dtype=torch.float64)
    duration[..., 1] = 0.0
    transition = torch.zeros(4, 1, 2, 1, dtype=torch.float64)
    emission = torch.log(torch.tensor([[0.8], [0.5], [0.25], [0.4]], dtype=torch.float64))

    evidence = causal_prefix_log_evidence(initial, duration, transition, emission)
    expected = torch.cumsum(emission[:, 0], dim=0)
    torch.testing.assert_close(evidence, expected, atol=1e-12, rtol=1e-12)

    end_probability = causal_prefix_episode_end_probability(initial, duration, transition, emission)
    torch.testing.assert_close(
        end_probability,
        torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
