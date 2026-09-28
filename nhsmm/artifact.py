from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import torch

from nhsmm.config import ModelConfig
from nhsmm.context import ContextEncoder
from nhsmm.encoder import DefaultEncoder
from nhsmm.inference import load_inference_model, prepare_inference
from nhsmm.models import NHSMM

ARTIFACT_TYPE = "nhsmm"
ARTIFACT_VERSION = 1
_ARTIFACT_KEYS = {
    "artifact_type",
    "artifact_version",
    "model_config",
    "encoder_config",
    "schema",
    "state_dict",
}


def _raw_default_encoder(model: NHSMM) -> DefaultEncoder:
    encoder = model.encoder.encoder if isinstance(model.encoder, ContextEncoder) else model.encoder
    if not isinstance(encoder, DefaultEncoder):
        raise ValueError("artifact v1 supports only the canonical DefaultEncoder")
    return encoder


def _encoder_config(encoder: DefaultEncoder) -> dict[str, Any]:
    return {
        "type": "DefaultEncoder",
        "n_features": int(encoder.n_features),
        "cnn_kernel": int(encoder.cnn_kernel),
        "hidden_dim": int(encoder.hidden_dim),
        "cnn_channels": int(encoder.cnn_channels),
        "bidirectional": bool(encoder.bidirectional),
        "return_sequence": bool(encoder.return_sequence),
        "use_packed": bool(encoder.use_packed),
        "causal": bool(encoder.causal),
    }


def _schema(config: ModelConfig) -> dict[str, Any]:
    return {
        "n_states": int(config.n_states),
        "n_features": int(config.n_features),
        "max_duration": int(config.max_duration),
        "causal": bool(config.causal),
        "emission_type": str(config.emission_type),
        "transition_type": str(config.transition_type),
    }


def build_artifact(model: NHSMM) -> dict[str, Any]:
    """Build a versioned, self-describing artifact payload from an NHSMM."""

    prepare_inference(model, freeze=False)
    encoder = _raw_default_encoder(model)
    config = asdict(model.config)
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "model_config": config,
        "encoder_config": _encoder_config(encoder),
        "schema": _schema(model.config),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }


def save_artifact(model: NHSMM, path: Union[str, Path]) -> Path:
    """Persist a trusted NHSMM artifact using the versioned v1 payload."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_artifact(model), target)
    return target


def _require_mapping(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return payload


def _validate_payload(payload: Any) -> tuple[ModelConfig, Mapping[str, torch.Tensor]]:
    payload = _require_mapping(payload, "artifact")
    keys = set(payload)
    if keys != _ARTIFACT_KEYS:
        missing = sorted(_ARTIFACT_KEYS - keys)
        extra = sorted(keys - _ARTIFACT_KEYS)
        raise ValueError(f"artifact keys mismatch: missing={missing}, extra={extra}")
    if payload["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError(f"unsupported artifact type: {payload['artifact_type']!r}")
    if payload["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version: {payload['artifact_version']!r}")

    config_data = _require_mapping(payload["model_config"], "model_config")
    try:
        config = ModelConfig(**dict(config_data))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ModelConfig in artifact: {exc}") from exc

    schema = _require_mapping(payload["schema"], "schema")
    expected_schema = _schema(config)
    if dict(schema) != expected_schema:
        raise ValueError(
            f"artifact schema metadata does not match ModelConfig: "
            f"expected={expected_schema}, got={dict(schema)}"
        )

    encoder_meta = _require_mapping(payload["encoder_config"], "encoder_config")
    expected_encoder_keys = {
        "type",
        "n_features",
        "cnn_kernel",
        "hidden_dim",
        "cnn_channels",
        "bidirectional",
        "return_sequence",
        "use_packed",
        "causal",
    }
    if set(encoder_meta) != expected_encoder_keys:
        raise ValueError("encoder_config keys do not match DefaultEncoder artifact v1 schema")
    if encoder_meta["type"] != "DefaultEncoder":
        raise ValueError("artifact v1 supports only DefaultEncoder")
    if int(encoder_meta["n_features"]) != config.n_features:
        raise ValueError("encoder n_features does not match ModelConfig")
    if int(encoder_meta["cnn_kernel"]) != config.cnn_kernel:
        raise ValueError("encoder cnn_kernel does not match ModelConfig")
    if int(encoder_meta["cnn_channels"]) != config.cnn_channels:
        raise ValueError("encoder cnn_channels does not match ModelConfig")
    if bool(encoder_meta["causal"]) != config.causal:
        raise ValueError("encoder causal mode does not match ModelConfig")
    if bool(encoder_meta["bidirectional"]) != (not config.causal):
        raise ValueError("encoder directionality does not match ModelConfig.causal")
    encoder_output_dim = int(encoder_meta["hidden_dim"]) * (
        2 if bool(encoder_meta["bidirectional"]) else 1
    )
    if (
        config.hidden_dim is None
        or config.context_dim is None
        or config.hidden_dim != config.context_dim
        or encoder_output_dim != config.context_dim
    ):
        raise ValueError(
            "encoder output dimension does not match resolved ModelConfig context dimension"
        )

    state_dict = _require_mapping(payload["state_dict"], "state_dict")
    if not state_dict:
        raise ValueError("artifact state_dict must not be empty")
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("artifact state_dict must map string names to tensors")
    return config, state_dict


def load_artifact(
    path: Union[str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    require_causal: bool = False,
    freeze: bool = True,
) -> NHSMM:
    """Safely load and validate a versioned NHSMM artifact.

    ``weights_only=True`` keeps artifact deserialization limited to tensors and
    primitive container data. Semantic compatibility is then checked before the
    strict model state load.
    """

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to deserialize NHSMM artifact: {exc}") from exc
    config, state_dict = _validate_payload(payload)
    model = load_inference_model(
        config,
        state_dict,
        device=device,
        require_causal=require_causal,
        freeze=freeze,
    )
    actual_encoder = _encoder_config(_raw_default_encoder(model))
    if actual_encoder != dict(payload["encoder_config"]):
        raise ValueError(
            "reconstructed encoder does not match artifact encoder configuration"
        )
    return model
