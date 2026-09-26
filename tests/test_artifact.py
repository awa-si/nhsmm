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
