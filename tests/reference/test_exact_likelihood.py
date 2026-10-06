from __future__ import annotations

import torch

from tests.reference.hsmm_exact import (
    causal_log_likelihood,
    enumerate_causal_paths,
    enumerate_segment_paths,
    segment_log_likelihood,
)


def _fixture(dtype: torch.dtype = torch.float64):
    initial = torch.log(torch.tensor([0.7, 0.3], dtype=dtype))
    duration = torch.log(
        torch.tensor(
            [
                [[0.6, 0.4], [0.25, 0.75]],
                [[0.5, 0.5], [0.8, 0.2]],
                [[0.9, 0.1], [0.4, 0.6]],
            ],
            dtype=dtype,
        )
    )
    transition = torch.log(
        torch.tensor(
            [
                [
                    [[0.8, 0.2], [0.4, 0.6]],
                    [[0.3, 0.7], [0.9, 0.1]],
                ],
                [
                    [[0.65, 0.35], [0.2, 0.8]],
                    [[0.55, 0.45], [0.1, 0.9]],
                ],
                [
                    [[0.7, 0.3], [0.6, 0.4]],
                    [[0.4, 0.6], [0.25, 0.75]],
                ],
            ],
            dtype=dtype,
        )
    )
    emission = torch.log(torch.tensor([[0.9, 0.3], [0.5, 0.8], [0.7, 0.4]], dtype=dtype))
    return initial, duration, transition, emission


def test_exact_reference_hand_fixture_k1_d1() -> None:
    initial = torch.zeros(1, dtype=torch.float64)
    duration = torch.zeros(3, 1, 1, dtype=torch.float64)
    transition = torch.zeros(3, 1, 1, 1, dtype=torch.float64)
    emission = torch.log(torch.tensor([[0.8], [0.5], [0.25]], dtype=torch.float64))

    causal = causal_log_likelihood(enumerate_causal_paths(initial, duration, transition, emission))
    segment = segment_log_likelihood(
        enumerate_segment_paths(initial, duration, transition, emission)
    )
    expected = emission.sum()
    torch.testing.assert_close(causal, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(segment, expected, atol=1e-12, rtol=1e-12)


def test_exact_reference_distinguishes_causal_prefix_from_segment_end_likelihood() -> None:
    initial, duration, transition, emission = _fixture()
    duration[:] = duration[0]
    transition[:] = transition[0]

    causal = causal_log_likelihood(enumerate_causal_paths(initial, duration, transition, emission))
    segment = segment_log_likelihood(
        enumerate_segment_paths(initial, duration, transition, emission)
    )

    # Causal prefix likelihood includes paths whose active final episode can
    # continue beyond the observation horizon. Segment likelihood requires the
    # final segment to terminate exactly at T-1, so the two contracts differ.
    assert causal > segment
    torch.testing.assert_close(causal, torch.tensor(-1.4472591157749417, dtype=torch.float64))
    torch.testing.assert_close(segment, torch.tensor(-1.8872809473542982, dtype=torch.float64))


def test_exact_segment_likelihood_matches_manual_two_timestep_sum() -> None:
    initial = torch.zeros(1, dtype=torch.float64)
    duration = torch.log(
        torch.tensor(
            [
                [[0.4, 0.6]],
                [[0.3, 0.7]],
            ],
            dtype=torch.float64,
        )
    )
    transition = torch.zeros(2, 1, 2, 1, dtype=torch.float64)
    emission = torch.log(torch.tensor([[0.8], [0.5]], dtype=torch.float64))

    actual = segment_log_likelihood(
        enumerate_segment_paths(initial, duration, transition, emission)
    )
    # Complete segmentations are either one duration-2 episode ending at t=1,
    # or two duration-1 episodes ending at t=0 and t=1 respectively.
    expected_probability = 0.8 * 0.5 * (0.7 + 0.4 * 0.3)
    expected = torch.tensor(expected_probability, dtype=torch.float64).log()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
