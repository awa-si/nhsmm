"""External context contracts across ragged/empty sequences and strict configs."""

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


def model():
    cfg = ModelConfig(
        n_states=2, n_features=2, max_duration=3, causal=True, context_dim=3, seed=901, dropout=0.0
    )
    value = NHSMM(cfg, device="cpu")
    value.initialize_distributions()
    return value.double()


def test_ragged_context_likelihood_and_gradients_match_explicit_padding():
    value = model()
    x = [
        torch.randn(5, 2, dtype=torch.float64),
        torch.empty(0, 2),
        torch.randn(3, 2, dtype=torch.float64),
    ]
    contexts = [torch.randn(len(item), 3, dtype=torch.float64, requires_grad=True) for item in x]
    padded = torch.randn(3, 5, 3, dtype=torch.float64)
    with torch.no_grad():
        for index, item in enumerate(contexts):
            padded[index, : len(item)] = item
    padded.requires_grad_()
    a = value.log_likelihood(x, context=contexts)
    b = value.log_likelihood(x, context=padded)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    grad_a = torch.autograd.grad(a.sum(), contexts)
    grad_b = torch.autograd.grad(b.sum(), padded)[0]
    for index, item in enumerate(contexts):
        torch.testing.assert_close(grad_a[index], grad_b[index, : len(item)], rtol=0, atol=1e-12)
        assert torch.equal(grad_b[index, len(item) :], torch.zeros_like(grad_b[index, len(item) :]))


@pytest.mark.parametrize("form", ["global", "shared", "static", "full", "list"])
@pytest.mark.parametrize("batch", [0, 2])
def test_empty_external_context_has_valid_canonical_shape_and_neutral_likelihood(form, batch):
    value = model()
    x = torch.empty(batch, 0, 2)
    context = {
        "global": torch.zeros(3),
        "shared": torch.empty(0, 3),
        "static": torch.zeros(batch, 1, 3),
        "full": torch.empty(batch, 0, 3),
        "list": [torch.empty(0, 3) for _ in range(batch)],
    }[form]
    sequence = value._build_sequence_set(x, context=context)
    assert sequence.canonical.shape == (batch, 1, 3)
    assert value.forward(sequence).shape == (batch, 0, 2, 3)
    assert torch.equal(
        value.log_likelihood(x, context=context), torch.zeros(batch, dtype=torch.float64)
    )


@pytest.mark.parametrize("choice", [["mean"], {"mean": 1}, {"mean"}])
def test_non_string_config_choices_raise_validation_error(choice):
    with pytest.raises(ValueError, match="pool must be one of"):
        ModelConfig(n_states=2, n_features=2, pool=choice)


def test_config_parsing_strict_and_permissive_paths_share_validated_fields():
    payload = {"n_states": 2, "n_features": 2, "future": 99, 42: "unknown"}
    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        ModelConfig.from_dict(payload)
    result = ModelConfig.from_dict(payload, strict=False)
    assert result == ModelConfig(n_states=2, n_features=2)
    assert payload[42] == "unknown" and payload["future"] == 99
