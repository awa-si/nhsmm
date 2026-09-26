from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from nhsmm.filtering import (
    HSMMFilterState,
    filter_step,
    initialize_filter,
)
from nhsmm.survival import (
    HSMMSurvivalForecast,
    HorizonInput,
    active_episode_survival_forecast,
)
from nhsmm.transitions import HSMMTransitionForecast, one_step_transition_forecast


def _require_compatible(reference: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if tensor.device != reference.device:
        raise ValueError(f"{name} device {tensor.device} != expected {reference.device}")
    if tensor.dtype != reference.dtype:
        raise ValueError(f"{name} dtype {tensor.dtype} != expected {reference.dtype}")


@dataclass(frozen=True)
class HSMMRuntimeState:
    """Online causal filtering state kept separate from model parameters.

    The current encoder does not expose an incremental hidden-state API, so
    ``observations`` retains the processed prefix. Duration and transition
    scores are conditioned on the latest processed prefix and are consumed by
    the next boundary update.
    """

    filter_state: HSMMFilterState
    observations: torch.Tensor  # [B,T,F]
    duration_log_prob: torch.Tensor  # [B,K,D]
    transition_log_prob: torch.Tensor  # [B,K,D,K]
    last_timestamp: Optional[Any] = None
    uses_timestamps: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.filter_state, HSMMFilterState):
            raise TypeError("filter_state must be an HSMMFilterState")
        if not isinstance(self.observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if self.observations.ndim != 3 or self.observations.shape[1] < 1:
            raise ValueError(
                f"observations must be [B,T,F] with T>=1, got {self.observations.shape}"
            )
        if not self.observations.is_floating_point():
            raise TypeError("observations must use a floating dtype")

        posterior = self.filter_state.log_posterior
        B, K, D = posterior.shape
        if self.observations.shape[0] != B:
            raise ValueError("observation batch does not match filter state")
        _require_compatible(posterior, self.observations, "observations")

        for name, tensor, shape in (
            ("duration_log_prob", self.duration_log_prob, (B, K, D)),
            ("transition_log_prob", self.transition_log_prob, (B, K, D, K)),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {tensor.shape}")
            _require_compatible(posterior, tensor, name)

        if self.uses_timestamps and self.last_timestamp is None:
            raise ValueError("timestamped runtime state requires last_timestamp")
        if not self.uses_timestamps and self.last_timestamp is not None:
            raise ValueError("untimestamped runtime state must not carry last_timestamp")


def _validate_model(model: Any) -> None:
    config = getattr(model, "config", None)
    if not bool(getattr(config, "causal", False)):
        raise ValueError("causal model filtering requires ModelConfig.causal=True")
    if bool(getattr(model, "training", False)):
        raise RuntimeError("model must be in eval mode before causal filtering")
    if getattr(model, "dist", None) is None:
        raise RuntimeError("model distributions are not initialized")


def _as_observation(observation: torch.Tensor, *, n_features: int) -> torch.Tensor:
    if not isinstance(observation, torch.Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if not observation.is_floating_point():
        raise TypeError("observation must use a floating dtype")

    if observation.ndim == 1:
        observation = observation.view(1, 1, -1)
    elif observation.ndim == 2:
        observation = observation.unsqueeze(1)
    elif observation.ndim == 3 and observation.shape[1] == 1:
        pass
    else:
        raise ValueError(
            "observation must be [F], [B,F], or [B,1,F] for one streaming timestep"
        )

    if observation.shape[-1] != n_features:
        raise ValueError(
            f"feature dimension mismatch: expected {n_features}, got {observation.shape[-1]}"
        )
    return observation


def _boundary_scores(
    model: Any,
    context: torch.Tensor,
    *,
    temperature: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    duration = model.dist.duration.log_matrix(
        context=context,
        temperature=temperature,
        T=1,
        soft_dmax=model.duration_logits_bias,
    )[:, 0]
    transition = model.dist.transition.log_matrix(
        context=context,
        temperature=temperature,
        T=1,
        soft_dmax=model.duration_logits_bias,
    )[:, 0]

    B = context.shape[0]
    K = int(model.config.n_states)
    D = int(model.dist.duration.max_duration)
    if duration.shape != (B, K, D):
        raise ValueError(f"duration logits must be {(B, K, D)}, got {duration.shape}")
    if transition.shape != (B, K, D, K):
        raise ValueError(
            f"transition logits must be {(B, K, D, K)}, got {transition.shape}"
        )
    return duration, transition


class HSMMFilterRuntime:
    """Correctness-first live causal HSMM filter.

    ``step`` is semantically incremental, but until the encoder exposes its own
    incremental state it re-encodes the retained causal observation prefix.
    The HSMM posterior itself is advanced exactly once per accepted timestamp.
    """

    def __init__(self, model: Any, temperature: Optional[float] = None) -> None:
        self.model = model
        self.temperature = temperature
        self.state: Optional[HSMMRuntimeState] = None

    def reset(self) -> None:
        self.state = None

    @torch.inference_mode()
    def forecast_survival(self, horizons: HorizonInput) -> HSMMSurvivalForecast:
        """Forecast current-episode survival using only the current information set."""

        if self.state is None:
            raise RuntimeError("runtime must process at least one observation before forecasting")
        return active_episode_survival_forecast(
            self.state.filter_state,
            self.state.duration_log_prob,
            horizons,
        )

    @torch.inference_mode()
    def forecast_transition(self) -> HSMMTransitionForecast:
        """Forecast the next boundary/state transition using only the current information set."""

        if self.state is None:
            raise RuntimeError("runtime must process at least one observation before forecasting")
        return one_step_transition_forecast(
            self.state.filter_state,
            self.state.duration_log_prob,
            self.state.transition_log_prob,
        )

    @torch.inference_mode()
    def step(
        self,
        observation: torch.Tensor,
        *,
        timestamp: Optional[Any] = None,
    ) -> HSMMFilterState:
        _validate_model(self.model)
        obs = _as_observation(
            observation,
            n_features=int(self.model.config.n_features),
        )

        if self.state is None:
            sequence = self.model._build_sequence_set(obs)
            B, T, K = sequence.log_probs.shape
            if T != 1:
                raise ValueError("runtime initialization must produce exactly one timestep")

            initial = self.model.dist.initial.log_matrix(
                context=sequence.canonical,
                temperature=self.temperature,
                T=1,
            )
            if initial.shape != (B, 1, K):
                raise ValueError(
                    f"initial logits must be {(B, 1, K)}, got {initial.shape}"
                )

            filter_state = initialize_filter(
                initial[:, 0],
                sequence.log_probs[:, 0],
                int(self.model.dist.duration.max_duration),
            )
            duration, transition = _boundary_scores(
                self.model,
                sequence.contexts[:, -1:],
                temperature=self.temperature,
            )
            self.state = HSMMRuntimeState(
                filter_state=filter_state,
                observations=sequence.sequences.clone(),
                duration_log_prob=duration,
                transition_log_prob=transition,
                last_timestamp=timestamp,
                uses_timestamps=timestamp is not None,
            )
            return filter_state

        previous = self.state
        if previous.uses_timestamps:
            if timestamp is None:
                raise ValueError("timestamp is required after timestamped initialization")
            try:
                if timestamp <= previous.last_timestamp:
                    raise ValueError("timestamp must be strictly increasing")
            except TypeError as exc:
                raise TypeError("timestamps must support strict ordering") from exc
        elif timestamp is not None:
            raise ValueError(
                "timestamp mode is fixed at initialization; reset before enabling timestamps"
            )

        if obs.shape[0] != previous.observations.shape[0]:
            raise ValueError("streaming batch size cannot change without reset")
        _require_compatible(previous.observations, obs, "observation")

        observations = torch.cat((previous.observations, obs), dim=1)
        sequence = self.model._build_sequence_set(observations)
        expected_lengths = torch.full_like(sequence.lengths, observations.shape[1])
        if not torch.equal(sequence.lengths, expected_lengths):
            raise ValueError(
                "runtime filtering requires one valid observation per batch item"
            )

        filter_state = filter_step(
            previous.filter_state,
            sequence.log_probs[:, -1],
            previous.duration_log_prob,
            previous.transition_log_prob,
        )
        duration, transition = _boundary_scores(
            self.model,
            sequence.contexts[:, -1:],
            temperature=self.temperature,
        )
        self.state = HSMMRuntimeState(
            filter_state=filter_state,
            observations=sequence.sequences.clone(),
            duration_log_prob=duration,
            transition_log_prob=transition,
            last_timestamp=timestamp if previous.uses_timestamps else None,
            uses_timestamps=previous.uses_timestamps,
        )
        return filter_state
