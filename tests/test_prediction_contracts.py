from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM


def test_decode_accepts_variable_length_sequence_list() -> None:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=211,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()

    sequences = [torch.randn(6, 4), torch.randn(4, 4)]
    decoded = model.decode(sequences, first_only=False, verbose=False)

    assert isinstance(decoded, list)
    assert [tuple(path.shape) for path in decoded] == [(6,), (4,)]
