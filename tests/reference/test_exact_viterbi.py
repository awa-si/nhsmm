from __future__ import annotations

import torch

from tests.reference.hsmm_exact import enumerate_segment_paths, segment_map_state_path


def test_exact_segment_map_preserves_duration_for_destination_transition() -> None:
    initial = torch.log(torch.tensor([1.0, 1e-30], dtype=torch.float64))
    duration = torch.log(
        torch.tensor(
            [
                [[1.0, 1e-30], [1.0, 1e-30]],
                [[0.525, 0.475], [1.0, 1e-30]],
                [[1.0, 1e-30], [1.0, 1e-30]],
            ],
            dtype=torch.float64,
        )
    )

    transition = torch.full((3, 2, 2, 2), -100.0, dtype=torch.float64)
    transition[0, 0, 0, 0] = 0.0
    transition[1, 0, 0] = torch.tensor([0.0, -10.0], dtype=torch.float64)
    transition[1, 0, 1] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    emission = torch.zeros(3, 2, dtype=torch.float64)
    emission[2, 0] = -5.0

    paths = enumerate_segment_paths(initial, duration, transition, emission)
    state_path, _ = segment_map_state_path(paths)
    assert state_path == [0, 0, 1]


def test_exact_causal_map_returns_state_and_age_path() -> None:
    from tests.reference.hsmm_exact import causal_map_state_age_path, enumerate_causal_paths

    initial = torch.tensor([0.0, float("-inf")], dtype=torch.float64)
    duration = torch.full((3, 2, 2), float("-inf"), dtype=torch.float64)
    duration[..., 1] = 0.0
    transition = torch.full((3, 2, 2, 2), float("-inf"), dtype=torch.float64)
    transition[:, 0, :, 1] = 0.0
    transition[:, 1, :, 0] = 0.0
    emission = torch.zeros(3, 2, dtype=torch.float64)

    paths = enumerate_causal_paths(initial, duration, transition, emission)
    states, ages, score = causal_map_state_age_path(paths)
    assert states == [0, 0, 1]
    assert ages == [1, 2, 1]
    assert torch.isfinite(score)
