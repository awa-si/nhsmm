from __future__ import annotations

import numpy as np
import pytest
import torch

from nhsmm.data import SequenceDataset, generate_gaussian_sequence


def test_generate_gaussian_sequence_is_deterministic() -> None:
    first = generate_gaussian_sequence(
        n_states=3,
        n_features=2,
        seed=17,
        n_segments_per_state=2,
        context_dim=2,
    )
    second = generate_gaussian_sequence(
        n_states=3,
        n_features=2,
        seed=17,
        n_segments_per_state=2,
        context_dim=2,
    )

    first_states, first_x, first_context = first
    second_states, second_x, second_context = second

    assert np.array_equal(first_states, second_states)
    assert torch.equal(first_x, second_x)
    assert first_context is not None
    assert second_context is not None
    assert torch.equal(first_context, second_context)


def test_generate_gaussian_sequence_does_not_mutate_torch_rng() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(4)

    torch.manual_seed(1234)
    generate_gaussian_sequence(
        n_states=2,
        n_features=2,
        seed=99,
        n_segments_per_state=1,
    )
    actual = torch.rand(4)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_states": 0}, "n_states"),
        ({"n_features": 0}, "n_features"),
        ({"n_segments_per_state": 0}, "n_segments_per_state"),
        ({"seg_len_range": (0, 3)}, "seg_len_range"),
        ({"seg_len_range": (3, 3)}, "seg_len_range"),
        ({"noise_scale": -0.1}, "noise_scale"),
        ({"context_noise_scale": float("nan")}, "context_noise_scale"),
        ({"context_dim": 0}, "context_dim"),
    ],
)
def test_generate_gaussian_sequence_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    base = {
        "n_states": 2,
        "n_features": 2,
        "seed": 7,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        generate_gaussian_sequence(**base)


def test_sequence_dataset_fixed_length_and_context_shapes() -> None:
    dataset = SequenceDataset(
        n_states=2,
        n_features=3,
        seed=5,
        n_segments_per_state=1,
        context_dim=4,
    )

    observation, context, state = dataset[0]

    assert observation.shape == (3,)
    assert context.shape == (4,)
    assert state.ndim == 0
    assert len(dataset) > 0


def test_sequence_dataset_loader_validates_arguments() -> None:
    dataset = SequenceDataset(
        n_states=2,
        n_features=2,
        seed=5,
        n_segments_per_state=1,
    )

    with pytest.raises(ValueError, match="batch_size"):
        dataset.loader(batch_size=0)
    with pytest.raises(ValueError, match="shuffle"):
        dataset.loader(shuffle=1)
