from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from nhsmm.filtering import filter_model_sequence


@dataclass(frozen=True)
class ModelHealthThresholds:
    """Thresholds used to classify fitted-model degeneration."""

    min_effective_states: float = 2.0
    max_state_occupancy: float = 0.80
    min_viterbi_states: int = 2
    min_state_occupancy: float = 0.01
    max_duration_peak: float = 0.95
    max_transition_peak: float = 0.95

    def __post_init__(self) -> None:
        if self.min_effective_states <= 0.0:
            raise ValueError("min_effective_states must be > 0")
        if not 0.0 < self.max_state_occupancy <= 1.0:
            raise ValueError("max_state_occupancy must be in (0,1]")
        if self.min_viterbi_states < 1:
            raise ValueError("min_viterbi_states must be >= 1")
        if not 0.0 <= self.min_state_occupancy < 1.0:
            raise ValueError("min_state_occupancy must be in [0,1)")
        if not 0.0 < self.max_duration_peak <= 1.0:
            raise ValueError("max_duration_peak must be in (0,1]")
        if not 0.0 < self.max_transition_peak <= 1.0:
            raise ValueError("max_transition_peak must be in (0,1]")


@dataclass(frozen=True)
class ModelHealthReport:
    """Aggregate model-health diagnostics over a batch of sequences."""

    occupancy: torch.Tensor
    viterbi_usage: torch.Tensor
    effective_states: float
    max_occupancy: float
    min_occupancy: float
    viterbi_states_used: int
    posterior_entropy_mean: float
    duration_entropy_mean: float
    duration_peak_mean: float
    duration_peak_max: float
    transition_entropy_mean: float
    transition_peak_mean: float
    transition_peak_max: float
    state_collapsed: bool
    dead_state: bool
    duration_degenerate: bool
    transition_degenerate: bool

    @property
    def healthy(self) -> bool:
        return not (self.state_collapsed or self.duration_degenerate or self.transition_degenerate)

    def as_dict(self) -> dict[str, object]:
        return {
            "occupancy": self.occupancy.tolist(),
            "viterbi_usage": self.viterbi_usage.tolist(),
            "effective_states": self.effective_states,
            "max_occupancy": self.max_occupancy,
            "min_occupancy": self.min_occupancy,
            "viterbi_states_used": self.viterbi_states_used,
            "posterior_entropy_mean": self.posterior_entropy_mean,
            "duration_entropy_mean": self.duration_entropy_mean,
            "duration_peak_mean": self.duration_peak_mean,
            "duration_peak_max": self.duration_peak_max,
            "transition_entropy_mean": self.transition_entropy_mean,
            "transition_peak_mean": self.transition_peak_mean,
            "transition_peak_max": self.transition_peak_max,
            "state_collapsed": self.state_collapsed,
            "dead_state": self.dead_state,
            "duration_degenerate": self.duration_degenerate,
            "transition_degenerate": self.transition_degenerate,
            "healthy": self.healthy,
        }


def _entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
    tiny = torch.finfo(probabilities.dtype).tiny
    safe = probabilities.clamp_min(tiny)
    return -(probabilities * safe.log()).sum(dim=dim)


def _flatten_viterbi(
    decoded: Union[torch.Tensor, list[torch.Tensor]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(decoded, list):
        if not decoded:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(
            [value.to(device=device, dtype=torch.long).reshape(-1) for value in decoded],
            dim=0,
        )
    return decoded.to(device=device, dtype=torch.long).reshape(-1)


def evaluate_model_health(
    model: Any,
    X: Union[torch.Tensor, list[torch.Tensor]],
    context: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    *,
    thresholds: Optional[ModelHealthThresholds] = None,
) -> ModelHealthReport:
    """Evaluate fitted NHSMM state usage and distribution degeneration.

    The function is observational: it restores the model's original train/eval
    mode after evaluation. Metrics are computed only over valid (unpadded)
    timesteps and use the actual aligned context supplied to the model.
    """

    thresholds = thresholds or ModelHealthThresholds()
    if getattr(model, "dist", None) is None:
        raise RuntimeError("model distributions are not initialized")

    original_training = bool(model.training)
    try:
        model.eval()
        with torch.inference_mode():
            _, input_mask = model._ensure_tensor(X, return_mask=True)
            if input_mask is None or int(input_mask.sum().item()) == 0:
                raise ValueError("model health requires at least one valid timestep")

            sequence = model._build_sequence_set(X, context=context)
            valid = sequence.masks.squeeze(-1)
            n_valid = int(valid.sum().item())

            trace = filter_model_sequence(model, X, context=context)
            state_posterior = trace.state_posterior
            valid_posterior = state_posterior[valid]

            occupancy = valid_posterior.mean(dim=0)
            occupancy = occupancy / occupancy.sum()
            occupancy_entropy = _entropy(occupancy, dim=0)
            effective_states = float(occupancy_entropy.exp().item())
            max_occupancy = float(occupancy.max().item())
            min_occupancy = float(occupancy.min().item())
            posterior_entropy_mean = float(_entropy(valid_posterior, dim=-1).mean().item())

            decoded = model.predict(X, context=context, mode="viterbi", verbose=False)
            flat_decoded = _flatten_viterbi(decoded, device=occupancy.device)
            if flat_decoded.numel() != n_valid:
                raise RuntimeError(
                    "Viterbi output length does not match the number of valid timesteps"
                )
            n_states = int(model.config.n_states)
            counts = torch.bincount(flat_decoded, minlength=n_states).to(occupancy.dtype)
            viterbi_usage = counts / counts.sum()
            viterbi_states_used = int((viterbi_usage >= thresholds.min_state_occupancy).sum())

            B, T = valid.shape
            duration_log = model.dist.duration.log_matrix(
                context=sequence.contexts,
                T=T,
                soft_dmax=model.duration_logits_bias,
            )
            transition_log = model.dist.transition.log_matrix(
                context=sequence.contexts,
                T=T,
                soft_dmax=model.duration_logits_bias,
            )
            if duration_log.shape[:2] != (B, T):
                raise RuntimeError("duration diagnostics received incompatible shape")
            if transition_log.shape[:2] != (B, T):
                raise RuntimeError("transition diagnostics received incompatible shape")

            duration_prob = duration_log.exp()[valid]
            duration_entropy = _entropy(duration_prob, dim=-1)
            duration_peak = duration_prob.max(dim=-1).values

            transition_prob = transition_log.exp()[valid]
            transition_entropy = _entropy(transition_prob, dim=-1)
            transition_peak = transition_prob.max(dim=-1).values

            duration_entropy_mean = float(duration_entropy.mean().item())
            duration_peak_mean = float(duration_peak.mean().item())
            duration_peak_max = float(duration_peak.max().item())
            transition_entropy_mean = float(transition_entropy.mean().item())
            transition_peak_mean = float(transition_peak.mean().item())
            transition_peak_max = float(transition_peak.max().item())

        effective_floor = min(float(n_states), thresholds.min_effective_states)
        required_viterbi = min(n_states, thresholds.min_viterbi_states)
        state_collapsed = (
            effective_states < effective_floor
            or max_occupancy > thresholds.max_state_occupancy
            or viterbi_states_used < required_viterbi
        )
        dead_state = min_occupancy < thresholds.min_state_occupancy
        duration_degenerate = duration_peak_max > thresholds.max_duration_peak
        transition_degenerate = transition_peak_max > thresholds.max_transition_peak

        return ModelHealthReport(
            occupancy=occupancy.detach().cpu(),
            viterbi_usage=viterbi_usage.detach().cpu(),
            effective_states=effective_states,
            max_occupancy=max_occupancy,
            min_occupancy=min_occupancy,
            viterbi_states_used=viterbi_states_used,
            posterior_entropy_mean=posterior_entropy_mean,
            duration_entropy_mean=duration_entropy_mean,
            duration_peak_mean=duration_peak_mean,
            duration_peak_max=duration_peak_max,
            transition_entropy_mean=transition_entropy_mean,
            transition_peak_mean=transition_peak_mean,
            transition_peak_max=transition_peak_max,
            state_collapsed=state_collapsed,
            dead_state=dead_state,
            duration_degenerate=duration_degenerate,
            transition_degenerate=transition_degenerate,
        )
    finally:
        model.train(original_training)
