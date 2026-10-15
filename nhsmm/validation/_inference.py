from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


@contextmanager
def _evaluation_mode(model: Any) -> Iterator[None]:
    """Temporarily evaluate, preserving every PyTorch submodule's mode."""
    if isinstance(model, torch.nn.Module):
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            yield
        finally:
            # train(flag) recurses and would erase deliberately mixed modes.
            for module, training in modes:
                module.training = training
    else:
        training = bool(getattr(model, "training", False))
        try:
            if hasattr(model, "eval"):
                model.eval()
            yield
        finally:
            if hasattr(model, "train"):
                model.train(training)
