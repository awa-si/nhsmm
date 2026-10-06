from __future__ import annotations

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.artifact import build_artifact, load_artifact, save_artifact
from nhsmm.filtering import filter_model_sequence


def _model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        use_context_encoder=True,
        dropout=0.0,
        seed=41,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_artifact_round_trip_preserves_inference(tmp_path) -> None:
    torch.manual_seed(43)
    source = _model()
    x = torch.randn(1, 6, source.config.n_features)
    expected = filter_model_sequence(source, x).log_posterior

    path = save_artifact(source, tmp_path / "model.nhsmm.pt")
    loaded = load_artifact(path, device="cpu", require_causal=True)
    actual = filter_model_sequence(loaded, x).log_posterior

    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.config == source.config
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_artifact_rejects_unknown_version_and_schema_mismatch(tmp_path) -> None:
    source = _model()
    payload = build_artifact(source)

    payload["artifact_version"] = 999
    version_path = tmp_path / "bad-version.pt"
    torch.save(payload, version_path)
    with pytest.raises(ValueError, match="unsupported artifact version"):
        load_artifact(version_path, require_causal=True)

    payload = build_artifact(source)
    payload["schema"] = dict(payload["schema"])
    payload["schema"]["n_features"] += 1
    schema_path = tmp_path / "bad-schema.pt"
    torch.save(payload, schema_path)
    with pytest.raises(ValueError, match="schema metadata"):
        load_artifact(schema_path, require_causal=True)


def test_artifact_rejects_causal_metadata_mismatch(tmp_path) -> None:
    source = _model()
    payload = build_artifact(source)
    payload["encoder_config"] = dict(payload["encoder_config"])
    payload["encoder_config"]["causal"] = False
    path = tmp_path / "bad-encoder.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="encoder causal mode"):
        load_artifact(path, require_causal=True)


def test_artifact_round_trip_with_explicit_context_dim(tmp_path) -> None:
    torch.manual_seed(47)
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        use_context_encoder=True,
        context_dim=2,
        dropout=0.0,
        seed=47,
    )
    source = NHSMM(config, device="cpu")
    source.initialize_distributions()
    source.eval()
    x = torch.randn(2, 5, config.n_features)
    expected = filter_model_sequence(source, x).log_posterior

    path = save_artifact(source, tmp_path / "context-model.nhsmm.pt")
    loaded = load_artifact(path, device="cpu", require_causal=True)
    actual = filter_model_sequence(loaded, x).log_posterior

    assert loaded.config.context_dim == 2
    assert loaded.config.hidden_dim == 2
    assert loaded.encoder.encoder.hidden_dim == 2
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_noncausal_artifact_round_trip_preserves_explicit_context_dim(tmp_path) -> None:
    config = ModelConfig(
        n_states=2,
        n_features=3,
        max_duration=4,
        causal=False,
        use_context_encoder=True,
        context_dim=2,
        hidden_dim=2,
        dropout=0.0,
        seed=53,
    )
    source = NHSMM(config, device="cpu")
    source.initialize_distributions()
    source.eval()
    x = torch.randn(1, 4, config.n_features)
    expected = source.log_likelihood(x)

    path = save_artifact(source, tmp_path / "noncausal-context-model.nhsmm.pt")
    loaded = load_artifact(path, device="cpu")
    actual = loaded.log_likelihood(x)

    assert loaded.encoder.encoder.hidden_dim == 1
    assert loaded.encoder.encoder.out_dim == 2
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_build_artifact_is_observational() -> None:
    source = _model()
    source.train()
    for parameter in source.parameters():
        parameter.requires_grad_(True)
    training_before = source.training
    requires_grad_before = {name: p.requires_grad for name, p in source.named_parameters()}

    payload = build_artifact(source)

    assert payload["artifact_type"] == "nhsmm"
    assert source.training is training_before
    assert {name: p.requires_grad for name, p in source.named_parameters()} == requires_grad_before


def test_build_artifact_rejects_uninitialized_model_without_mutation() -> None:
    config = ModelConfig(n_states=2, n_features=2, seed=59)
    source = NHSMM(config, device="cpu")
    source.train()

    with pytest.raises(RuntimeError, match="distributions must be initialized"):
        build_artifact(source)

    assert source.training


def test_artifact_model_config_excludes_removed_unused_fields() -> None:
    source = _model()
    payload = build_artifact(source)

    assert "temperature" not in payload["model_config"]
    assert "debug" not in payload["model_config"]


def test_build_artifact_rejects_nonfinite_persistent_buffer() -> None:
    source = _model()
    source.register_buffer("poisoned_buffer", torch.tensor(float("inf")), persistent=True)

    with pytest.raises(ValueError, match="poisoned_buffer"):
        build_artifact(source)


def test_load_artifact_rejects_nonfinite_state_tensor(tmp_path) -> None:
    source = _model()
    payload = build_artifact(source)
    name = next(key for key, value in payload["state_dict"].items() if value.is_floating_point())
    payload["state_dict"][name] = payload["state_dict"][name].clone()
    payload["state_dict"][name].reshape(-1)[0] = float("nan")
    path = tmp_path / "nonfinite.nhsmm.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="invalid artifact state_dict"):
        load_artifact(path, device="cpu")
