from __future__ import annotations

import types

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


def test_log_likelihood_rejects_positive_infinity() -> None:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=223,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()

    def fake_forward(self, seq_set, context=None):
        batch, timesteps, _ = seq_set.sequences.shape
        return torch.full(
            (batch, timesteps, self.config.n_states, self.config.max_duration),
            float("inf"),
        )

    model.forward = types.MethodType(fake_forward, model)

    with pytest.raises(ValueError, match=r"log_likelihood produced \+inf"):
        model.log_likelihood(torch.randn(1, 3, 4), reduce=False)
