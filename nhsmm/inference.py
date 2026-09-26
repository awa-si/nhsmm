from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Union

import torch

from nhsmm.config import ModelConfig
from nhsmm.models import NHSMM


def prepare_inference(model: NHSMM, *, freeze: bool = True) -> NHSMM:
    """Validate and prepare an initialized NHSMM for inference.

    This function does not deserialize artifacts. It establishes the runtime
    contract after construction/loading: distributions must be initialized,
    all floating model parameters must be finite, the model is switched to
    eval mode, and parameters are frozen by default to prevent accidental
    training-time mutation in production inference code.
    """

    if not isinstance(model, NHSMM):
        raise TypeError("model must be an NHSMM")
    if model.dist is None:
        raise RuntimeError("model distributions must be initialized before inference")

    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and not torch.isfinite(parameter).all():
            raise ValueError(f"model parameter {name!r} contains NaN or infinity")

    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model


def load_inference_model(
    config: ModelConfig,
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: Optional[Union[str, torch.device]] = None,
    require_causal: bool = False,
    freeze: bool = True,
) -> NHSMM:
    """Construct and strictly load an inference-ready NHSMM.

    ``state_dict`` must already be deserialized by the caller. Artifact format,
    versioning, provenance, and metadata validation belong to the artifact
    contract and are intentionally kept separate from this model-loading path.
    """

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    if require_causal and not config.causal:
        raise ValueError("production causal inference requires ModelConfig.causal=True")
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping of parameter names to tensors")
    if not state_dict:
        raise ValueError("state_dict must not be empty")
    for name, value in state_dict.items():
        if not isinstance(name, str):
            raise TypeError("state_dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a torch.Tensor")

    model = NHSMM(config=config, device=device)
    model.initialize_distributions()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"incompatible NHSMM state_dict: {exc}") from exc

    return prepare_inference(model, freeze=freeze)
