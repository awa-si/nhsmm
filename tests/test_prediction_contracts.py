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


def test_log_likelihood_variable_length_batch_matches_individual_sequences() -> None:
    for causal in (True, False):
        config = ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=4,
            causal=causal,
            dropout=0.0,
            seed=227,
        )
        model = NHSMM(config, device="cpu")
        model.initialize_distributions()
        model.eval()

        first = torch.randn(7, 2)
        second = torch.randn(3, 2)
        batch = model.log_likelihood([first, second], reduce=False)
        separate = torch.stack(
            [
                model.log_likelihood(first, reduce=False)[0],
                model.log_likelihood(second, reduce=False)[0],
            ]
        )

        assert torch.allclose(batch, separate, atol=1e-5, rtol=1e-5)


def test_external_context_is_not_applied_twice_in_public_likelihood() -> None:
    config = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=4,
        causal=True,
        context_dim=4,
        hidden_dim=4,
        dropout=0.0,
        seed=229,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()

    observations = torch.randn(1, 5, 2)
    context = torch.randn(1, 5, 4)

    sequence = model._build_sequence_set(observations, context=context)
    alpha = model.forward(sequence)
    expected = torch.logsumexp(alpha[0, -1].flatten(), dim=0)
    actual = model.log_likelihood(observations, context=context, reduce=False)[0]

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
