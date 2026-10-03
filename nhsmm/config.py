from __future__ import annotations
from typing import Optional, Literal
from dataclasses import dataclass
import logging

import torch

logger = logging.getLogger("NHSMM")

if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


EPS: float = 1e-12
DTYPE = torch.float32
NEG_INF = torch.finfo(DTYPE).min
MIN_LOGITS: float = NEG_INF
MAX_LOGITS: float = 1e5


@dataclass
class ModelConfig:
    # -----------------------------
    # Model architecture
    # -----------------------------
    n_states: int
    n_features: int
    pad_value: float = 0.0

    # -----------------------------
    # Encoder
    # -----------------------------
    n_heads: int = 4
    cnn_kernel: int = 3
    cnn_channels: int = 5
    dropout: float = 0.05
    hidden_dim: Optional[int] = None
    context_dim: Optional[int] = None
    pool: Literal["mean", "last", "max", "attn", "mha"] = "mean"
    causal: bool = False

    # -----------------------------
    # HMM / Distribution
    # -----------------------------
    max_duration: int = 35
    min_covar: float = 1e-6
    temperature: float = 1.0
    emission_init_mode: Literal["random", "spread", "kmeans"] = "spread"
    initial_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    duration_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_context_max_delta: float = 0.5
    transition_type: Literal["ergodic", "semi", "left-to-right"] = "ergodic"
    activation: Literal["leaky_relu", "identity", "softplus", "gelu", "relu", "tanh"] = "tanh"
    emission_type: Literal["gaussian", "studentt"] = "gaussian"

    # -----------------------------
    # General
    # -----------------------------
    seed: Optional[int] = None
    debug: bool = False

    # -----------------------------
    # Optimization / Training
    # -----------------------------
    n_init: int = 3
    lr: float = 1e-2
    tol: float = 1e-4
    max_iter: int = 40
    transition_refine_steps: int = 0
    transition_refine_lr: float = 3e-2
    loss_bias: float = 1e-3
    plateau_window: int = 6
    plateau_tol: float = 1e-4
    use_scheduler: bool = True
    convergence_stop: bool = True
    convergence_mode: Literal["delta", "plateau"] = "plateau"
    verbose: bool = True

    def __post_init__(self) -> None:
        integer_minimums = {
            "n_states": (self.n_states, 1),
            "n_features": (self.n_features, 1),
            "n_heads": (self.n_heads, 1),
            "cnn_kernel": (self.cnn_kernel, 1),
            "cnn_channels": (self.cnn_channels, 1),
            "max_duration": (self.max_duration, 1),
            "n_init": (self.n_init, 1),
            "max_iter": (self.max_iter, 1),
            "plateau_window": (self.plateau_window, 1),
            "transition_refine_steps": (self.transition_refine_steps, 0),
        }
        for name, (value, minimum) in integer_minimums.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        for name, value in (("hidden_dim", self.hidden_dim), ("context_dim", self.context_dim)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be None or an integer >= 1")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        for name, value in (
            ("min_covar", self.min_covar),
            ("temperature", self.temperature),
            ("lr", self.lr),
            ("transition_refine_lr", self.transition_refine_lr),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        for name, value in (
            ("transition_context_max_delta", self.transition_context_max_delta),
            ("tol", self.tol),
            ("loss_bias", self.loss_bias),
            ("plateau_tol", self.plateau_tol),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
