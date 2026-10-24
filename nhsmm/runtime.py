from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import math

import torch

from nhsmm.config import EPS
from nhsmm.context import (
    validate_masked_context,
    BASE_CONTEXT,
    CausalContext,
    align_context_tensor,
    align_context_availability,
    context_options,
)
from nhsmm.filtering import (
    HSMMFilterState,
    _filter_step_normalized,
    _validated_initial_prior,
    _initialize_prior_frontier,
    _require_compatible,
    initialize_filter,
)
from nhsmm.survival import (
    HSMMSurvivalForecast,
    HorizonInput,
    active_episode_survival_forecast,
)
from nhsmm.transitions import HSMMTransitionForecast, one_step_transition_forecast


class _RuntimeScoreCache:
    """Version-aware cache for static terms used by canonical runtime scorers."""

    def __init__(self) -> None:
        self._duration_key: Optional[tuple[int, int, torch.device, torch.dtype]] = None
        self._duration_gate_log: Optional[torch.Tensor] = None
        self._emission_key: Optional[tuple[int, int, torch.device, torch.dtype, float]] = None
        self._emission_var: Optional[torch.Tensor] = None
        self._emission_log_norm: Optional[torch.Tensor] = None

    def duration_gate_log(self, bias: torch.Tensor) -> torch.Tensor:
        key = (
            None
            if torch.is_inference(bias)
            else (id(bias), int(bias._version), bias.device, bias.dtype)
        )
        # Inference tensors have no version counter and can still be mutated
        # inside inference_mode. Recompute rather than reuse unprovable cache.
        if self._duration_gate_log is None or key is None or key != self._duration_key:
            with torch.no_grad():
                self._duration_gate_log = torch.sigmoid(bias).clamp_min(EPS).log()
            self._duration_key = key
        return self._duration_gate_log

    def gaussian_terms(
        self,
        log_var: torch.Tensor,
        *,
        min_covar: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            None
            if torch.is_inference(log_var)
            else (id(log_var), int(log_var._version), log_var.device, log_var.dtype, min_covar)
        )
        if self._emission_var is None or key is None or key != self._emission_key:
            with torch.no_grad():
                var = torch.nn.functional.softplus(log_var).clamp_min(min_covar)
                self._emission_var = var
                self._emission_log_norm = var.log() + math.log(2.0 * math.pi)
            self._emission_key = key
        assert self._emission_log_norm is not None
        return self._emission_var, self._emission_log_norm


def _validate_timestamp(timestamp: Any) -> None:
    if timestamp is None:
        return
    try:
        finite = math.isfinite(timestamp)
    except (TypeError, OverflowError):
        # Datetime-like and other scalar ordered objects need not support
        # conversion to float. Arbitrarily large integer timestamps are finite.
        finite = True
    if not finite:
        raise ValueError("timestamp must be finite and ordered")
    try:
        self_ordered = bool(timestamp <= timestamp)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("timestamp must support scalar ordering") from exc
    if not self_ordered:
        raise ValueError("timestamp must be finite and ordered")


def _is_default_distribution(value: Any, class_name: str) -> bool:
    cls = type(value)
    return cls.__module__ == "nhsmm.distributions.default" and cls.__name__ == class_name


@dataclass(frozen=True)
class HSMMRuntimeState:
    """Online causal filtering state kept separate from model parameters.

    For encoders that expose ``stream_step``, ``encoder_state`` carries only
    bounded causal CNN/LSTM state and ``observations`` stores the latest accepted
    timestep. Encoders without that contract fall back to retained-prefix
    re-encoding and keep the full observation prefix in ``observations``.
    """

    filter_state: HSMMFilterState
    observations: torch.Tensor  # [B,T,F], T==1 for incremental default encoder
    duration_log_prob: torch.Tensor  # [B,K,D]
    transition_log_prob: torch.Tensor  # [B,K,D,K]
    tail_end_probability: Optional[torch.Tensor] = None  # [B,K] in D+ mode
    encoder_state: Optional[Any] = None
    last_timestamp: Optional[Any] = None
    uses_timestamps: bool = False
    last_encoder_context: Optional[torch.Tensor] = None  # [B,1,H], for the next emission

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
        if not torch.isfinite(self.observations).all():
            raise ValueError("observations must contain only finite values")

        posterior = self.filter_state.log_posterior
        B, K, D = posterior.shape
        if self.observations.shape[0] != B:
            raise ValueError("observation batch does not match filter state")
        _require_compatible(posterior, self.observations, "observations")
        if self.last_encoder_context is not None:
            value = self.last_encoder_context
            if value.ndim != 3 or value.shape[:2] != (B, 1):
                raise ValueError("last_encoder_context must be [B,1,H]")
            _require_compatible(posterior, value, "last_encoder_context")
            if not torch.isfinite(value).all():
                raise ValueError("last_encoder_context must be finite")

        for name, tensor, shape in (
            ("duration_log_prob", self.duration_log_prob, (B, K, D)),
            ("transition_log_prob", self.transition_log_prob, (B, K, D, K)),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {tensor.shape}")
            _require_compatible(posterior, tensor, name)
            if torch.isnan(tensor).any() or torch.isposinf(tensor).any():
                raise ValueError(f"{name} must not contain NaN or +inf")
            if not torch.isfinite(torch.logsumexp(tensor, dim=-1)).all():
                raise ValueError(f"{name} must contain finite probability mass in every row")

        if self.tail_end_probability is not None:
            tail = self.tail_end_probability
            if tail.shape != (B, K):
                raise ValueError(f"tail_end_probability must be {(B, K)}, got {tail.shape}")
            _require_compatible(posterior, tail, "tail_end_probability")
            if not torch.isfinite(tail).all() or bool(((tail <= 0) | (tail > 1)).any()):
                raise ValueError("tail_end_probability must be finite and satisfy 0 < h <= 1")

        if self.uses_timestamps and self.last_timestamp is None:
            raise ValueError("timestamped runtime state requires last_timestamp")
        if not self.uses_timestamps and self.last_timestamp is not None:
            raise ValueError("untimestamped runtime state must not carry last_timestamp")
        _validate_timestamp(self.last_timestamp)


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
        raise ValueError("observation must be [F], [B,F], or [B,1,F] for one streaming timestep")

    if observation.shape[-1] != n_features:
        raise ValueError(
            f"feature dimension mismatch: expected {n_features}, got {observation.shape[-1]}"
        )
    if not torch.isfinite(observation).all():
        raise ValueError("observation must contain only finite values")
    return observation


def _boundary_scores(
    model: Any,
    context: Optional[torch.Tensor],
    *,
    temperature: Optional[float],
    cache: Optional[_RuntimeScoreCache] = None,
    batch_size: Optional[int] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_arg = context
    if context_arg is not None and context_arg.shape[-1] == 0:
        context_arg = None
    if context_arg is None:
        if batch_size is None:
            raise ValueError("batch_size is required when context is None")
        B = int(batch_size)
    else:
        B = context_arg.shape[0]
    options = context_options(context_mask)
    duration_dist = model.dist.duration
    duration_bias = model.duration_logits_bias
    if (
        cache is not None
        and _is_default_distribution(duration_dist, "Duration")
        and duration_bias.shape == (int(model.config.n_states), int(duration_dist.max_duration))
    ):
        duration_mod = duration_dist._modulate(
            context=context_arg,
            temperature=temperature,
            **options,
        )
        gate_log = cache.duration_gate_log(duration_bias)
        duration = torch.log_softmax(
            duration_mod + gate_log.view(1, 1, *gate_log.shape),
            dim=-1,
        )
        while duration.ndim < 4:
            duration = duration.unsqueeze(0)
        duration = duration[:, 0]
        if duration.shape[0] == 1 and B > 1:
            duration = duration.expand(B, *duration.shape[1:])
    else:
        duration = duration_dist.log_matrix(
            context=context_arg,
            temperature=temperature,
            **options,
            T=1,
            soft_dmax=duration_bias,
        )[:, 0]
        if duration.shape[0] == 1 and B > 1:
            duration = duration.expand(B, *duration.shape[1:])

    transition_dist = model.dist.transition
    if (
        cache is not None
        and _is_default_distribution(transition_dist, "Transition")
        and getattr(transition_dist, "transition_type", None) == "ergodic"
        and getattr(transition_dist, "max_duration", None) is not None
    ):
        base = transition_dist._tensor_shape(transition_dist.base)
        delta = transition_dist._masked_context_delta(base, context_arg, None, context_mask)
        transition_mod = transition_dist._validate_base(base + delta)
        transition_mod = transition_dist._apply_temperature(
            transition_mod,
            temperature,
        )
        transition_mod = transition_dist._apply_constraints(transition_mod)
        transition = torch.log_softmax(transition_mod, dim=-1)
        while transition.ndim < 5:
            transition = transition.unsqueeze(0)
        transition = transition[:, 0]
        if transition.shape[0] == 1 and B > 1:
            transition = transition.expand(B, *transition.shape[1:])
    else:
        transition = transition_dist.log_matrix(
            context=context_arg,
            temperature=temperature,
            **options,
            T=1,
            soft_dmax=duration_bias,
        )[:, 0]
        if transition.shape[0] == 1 and B > 1:
            transition = transition.expand(B, *transition.shape[1:])

    K = int(model.config.n_states)
    D = int(model.dist.duration.max_duration)
    if duration.shape != (B, K, D):
        raise ValueError(f"duration logits must be {(B, K, D)}, got {duration.shape}")
    if transition.shape != (B, K, D, K):
        raise ValueError(f"transition logits must be {(B, K, D, K)}, got {transition.shape}")
    return duration, transition


def _tail_end_probability(
    model: Any, batch_size: int, reference: torch.Tensor
) -> Optional[torch.Tensor]:
    fn = getattr(model.dist.duration, "tail_probability", None)
    tail = fn() if callable(fn) else None
    if tail is None:
        return None
    tail = tail.to(device=reference.device, dtype=reference.dtype)
    return tail.view(1, -1).expand(batch_size, -1)


def _stream_encoder(model: Any) -> Optional[Any]:
    context_encoder = getattr(model, "encoder", None)
    encoder = getattr(context_encoder, "encoder", None)
    if encoder is None:
        return None
    if not callable(getattr(encoder, "stream_step", None)):
        return None
    if not callable(getattr(encoder, "initial_stream_state", None)):
        return None
    return encoder


def _emission_log_prob(
    model: Any,
    observation: torch.Tensor,
    context: Optional[torch.Tensor],
    cache: Optional[_RuntimeScoreCache] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B = observation.shape[0]
    K = int(model.config.n_states)
    emission = model.dist.emission
    if (
        cache is not None
        and _is_default_distribution(emission, "Emission")
        and getattr(emission, "emission_type", None) == "gaussian"
    ):
        loc = emission._modulate(context=context, **context_options(context_mask))
        if loc.ndim == 2:
            loc = loc.unsqueeze(0).unsqueeze(0)
        var, log_norm = cache.gaussian_terms(
            emission.log_var,
            min_covar=float(emission.min_covar),
        )
        diff = observation[..., None, :] - loc
        log_prob = -0.5 * (diff.square() / var + log_norm).sum(dim=-1)
        if not torch.isfinite(log_prob).all():
            raise ValueError("emission log_prob contains NaN/Inf")
    else:
        log_prob = emission.log_prob(observation, context=context, **context_options(context_mask))
    if log_prob.shape != (B, 1, K):
        raise ValueError(f"emission log_prob must be {(B, 1, K)}, got {log_prob.shape}")
    return log_prob[:, 0]


class HSMMFilterRuntime:
    """Live causal HSMM filter with bounded encoder state when supported.

    The canonical ``DefaultEncoder(causal=True)`` exposes incremental causal
    CNN/LSTM state, so each accepted observation is encoded exactly once.
    Custom causal encoders without that API retain the correctness-first
    full-prefix fallback.
    """

    def __init__(
        self,
        model: Any,
        temperature: Optional[float] = None,
        *,
        initial_prior: Optional[HSMMFilterState] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._initial_prior = None
        if initial_prior is not None:
            if not isinstance(initial_prior, HSMMFilterState):
                raise TypeError("initial_prior must be HSMMFilterState")
            value = _validated_initial_prior(
                model, initial_prior, initial_prior.log_posterior.shape[0]
            )
            self._initial_prior = HSMMFilterState(value.detach().clone())
        self.state: Optional[HSMMRuntimeState] = None
        self._uses_external_context: Optional[bool] = None
        self._score_cache = _RuntimeScoreCache()

    def _initialize_filter(self, initial, emission, max_duration):
        if self._initial_prior is None:
            return initialize_filter(initial, emission, max_duration)
        value = _validated_initial_prior(self.model, self._initial_prior, emission.shape[0])
        return _initialize_prior_frontier(value, emission)

    def reset(self) -> None:
        self.state = None
        self._uses_external_context = None
        self._score_cache = _RuntimeScoreCache()

    @torch.inference_mode()
    def forecast_survival(self, horizons: HorizonInput) -> HSMMSurvivalForecast:
        """Forecast current-episode survival using only the current information set."""

        if self.state is None:
            raise RuntimeError("runtime must process at least one observation before forecasting")
        return active_episode_survival_forecast(
            self.state.filter_state,
            self.state.duration_log_prob,
            horizons,
            self.state.tail_end_probability,
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
            self.state.tail_end_probability,
        )

    @torch.inference_mode()
    def step(
        self,
        observation: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        timestamp: Optional[Any] = None,
    ) -> HSMMFilterState:
        _validate_model(self.model)
        _validate_timestamp(timestamp)
        obs = _as_observation(
            observation,
            n_features=int(self.model.config.n_features),
        )
        obs = obs.to(
            device=self.model.duration_logits_bias.device,
            dtype=self.model.duration_logits_bias.dtype,
        )
        encoder = _stream_encoder(self.model)
        uses_external_context = context is not None
        if (
            self._uses_external_context is not None
            and uses_external_context != self._uses_external_context
        ):
            raise ValueError(
                "context mode is fixed at initialization; reset before changing "
                "between internal and external context"
            )

        context_mask = None
        emission_context = None
        split_context = context if isinstance(context, CausalContext) else None
        if split_context is not None:
            validate_masked_context(self.model, split_context.available)
            context_mask = align_context_availability(
                split_context.available,
                lengths=torch.ones(obs.shape[0], device=obs.device, dtype=torch.long),
                timesteps=1,
                device=obs.device,
            )
            emission_context = align_context_tensor(
                split_context.emission,
                batch_size=obs.shape[0],
                timesteps=1,
                context_dim=int(self.model.context_dim or 0),
                device=obs.device,
                dtype=obs.dtype,
                available=context_mask,
            )
            context = split_context.boundary
        external_context = None
        if uses_external_context and context is not BASE_CONTEXT:
            if self.model.context_dim is None:
                raise ValueError(
                    "external context requires ModelConfig.context_dim to be configured"
                )
            external_context = align_context_tensor(
                context,
                batch_size=obs.shape[0],
                timesteps=1,
                context_dim=int(self.model.context_dim),
                device=obs.device,
                dtype=obs.dtype,
                available=context_mask,
            )

        if emission_context is None:
            emission_context = external_context

        if self.state is None:
            if uses_external_context:
                emission = _emission_log_prob(
                    self.model, obs, emission_context, self._score_cache, context_mask=context_mask
                )
                B = obs.shape[0]
                K = int(self.model.config.n_states)
                initial = self.model.dist.initial.log_matrix(
                    context=emission_context,
                    temperature=self.temperature,
                    T=1,
                    **context_options(context_mask, canonical=True),
                )
                if initial.shape[0] == 1 and B > 1:
                    initial = initial.expand(B, -1, -1)
                if initial.shape != (B, 1, K):
                    raise ValueError(f"initial logits must be {(B, 1, K)}, got {initial.shape}")
                filter_state = self._initialize_filter(
                    initial[:, 0],
                    emission,
                    int(self.model.dist.duration.max_duration),
                )
                duration, transition = _boundary_scores(
                    self.model,
                    external_context,
                    temperature=self.temperature,
                    cache=self._score_cache,
                    batch_size=B,
                    context_mask=context_mask,
                )
                self.state = HSMMRuntimeState(
                    filter_state=filter_state,
                    observations=obs.clone(),
                    duration_log_prob=duration,
                    transition_log_prob=transition,
                    tail_end_probability=_tail_end_probability(
                        self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                    ),
                    encoder_state=None,
                    last_timestamp=timestamp,
                    uses_timestamps=timestamp is not None,
                )
                self._uses_external_context = uses_external_context
                return filter_state

            if encoder is not None:
                encoder_state = encoder.initial_stream_state(
                    obs.shape[0],
                    device=obs.device,
                    dtype=obs.dtype,
                )
                context, encoder_state = encoder.stream_step(obs, encoder_state)
                start_context = torch.zeros_like(context)
                emission = _emission_log_prob(
                    self.model,
                    obs,
                    start_context,
                    self._score_cache,
                )
                B = context.shape[0]
                K = int(self.model.config.n_states)
                initial = self.model.dist.initial.log_matrix(
                    context=start_context,
                    temperature=self.temperature,
                    T=1,
                )
                if initial.shape != (B, 1, K):
                    raise ValueError(f"initial logits must be {(B, 1, K)}, got {initial.shape}")
                filter_state = self._initialize_filter(
                    initial[:, 0],
                    emission,
                    int(self.model.dist.duration.max_duration),
                )
                duration, transition = _boundary_scores(
                    self.model,
                    context,
                    temperature=self.temperature,
                    cache=self._score_cache,
                )
                self.state = HSMMRuntimeState(
                    filter_state=filter_state,
                    observations=obs.clone(),
                    duration_log_prob=duration,
                    transition_log_prob=transition,
                    tail_end_probability=_tail_end_probability(
                        self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                    ),
                    encoder_state=encoder_state,
                    last_encoder_context=context.clone(),
                    last_timestamp=timestamp,
                    uses_timestamps=timestamp is not None,
                )
                self._uses_external_context = uses_external_context
                return filter_state

            if getattr(self.model, "encoder", None) is None:
                B = obs.shape[0]
                K = int(self.model.config.n_states)
                emission = _emission_log_prob(
                    self.model,
                    obs,
                    None,
                    self._score_cache,
                )
                initial = self.model.dist.initial.log_matrix(
                    context=None,
                    temperature=self.temperature,
                    T=1,
                )
                if initial.shape[0] == 1 and B > 1:
                    initial = initial.expand(B, *initial.shape[1:])
                if initial.shape != (B, 1, K):
                    raise ValueError(f"initial logits must be {(B, 1, K)}, got {initial.shape}")
                filter_state = self._initialize_filter(
                    initial[:, 0],
                    emission,
                    int(self.model.dist.duration.max_duration),
                )
                duration, transition = _boundary_scores(
                    self.model,
                    None,
                    temperature=self.temperature,
                    cache=self._score_cache,
                    batch_size=B,
                )
                self.state = HSMMRuntimeState(
                    filter_state=filter_state,
                    observations=obs.clone(),
                    duration_log_prob=duration,
                    transition_log_prob=transition,
                    tail_end_probability=_tail_end_probability(
                        self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                    ),
                    encoder_state=None,
                    last_timestamp=timestamp,
                    uses_timestamps=timestamp is not None,
                )
                self._uses_external_context = uses_external_context
                return filter_state

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
                raise ValueError(f"initial logits must be {(B, 1, K)}, got {initial.shape}")

            filter_state = self._initialize_filter(
                initial[:, 0],
                sequence.log_probs[:, 0],
                int(self.model.dist.duration.max_duration),
            )
            duration, transition = _boundary_scores(
                self.model,
                sequence.contexts[:, -1:],
                temperature=self.temperature,
                cache=self._score_cache,
            )
            self.state = HSMMRuntimeState(
                filter_state=filter_state,
                observations=sequence.sequences.clone(),
                duration_log_prob=duration,
                transition_log_prob=transition,
                tail_end_probability=_tail_end_probability(
                    self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                ),
                encoder_state=None,
                last_timestamp=timestamp,
                uses_timestamps=timestamp is not None,
            )
            self._uses_external_context = uses_external_context
            return filter_state

        previous = self.state
        if previous.uses_timestamps:
            if timestamp is None:
                raise ValueError("timestamp is required after timestamped initialization")
            try:
                if not timestamp > previous.last_timestamp:
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

        if uses_external_context:
            emission = _emission_log_prob(
                self.model, obs, emission_context, self._score_cache, context_mask=context_mask
            )
            filter_state = _filter_step_normalized(
                previous.filter_state,
                emission,
                previous.duration_log_prob,
                previous.transition_log_prob,
                previous.tail_end_probability,
            )
            duration, transition = _boundary_scores(
                self.model,
                external_context,
                temperature=self.temperature,
                cache=self._score_cache,
                batch_size=obs.shape[0],
                context_mask=context_mask,
            )
            self.state = HSMMRuntimeState(
                filter_state=filter_state,
                observations=obs.clone(),
                duration_log_prob=duration,
                transition_log_prob=transition,
                tail_end_probability=_tail_end_probability(
                    self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                ),
                encoder_state=None,
                last_timestamp=timestamp if previous.uses_timestamps else None,
                uses_timestamps=previous.uses_timestamps,
            )
            self._uses_external_context = uses_external_context
            return filter_state

        if previous.encoder_state is not None:
            if encoder is None:
                raise RuntimeError("runtime encoder lost its incremental state contract")
            context, encoder_state = encoder.stream_step(obs, previous.encoder_state)
            if previous.last_encoder_context is None:
                raise RuntimeError("incremental runtime state lacks predictive emission context")
            emission = _emission_log_prob(
                self.model,
                obs,
                previous.last_encoder_context,
                self._score_cache,
            )
            filter_state = _filter_step_normalized(
                previous.filter_state,
                emission,
                previous.duration_log_prob,
                previous.transition_log_prob,
                previous.tail_end_probability,
            )
            duration, transition = _boundary_scores(
                self.model,
                context,
                temperature=self.temperature,
                cache=self._score_cache,
            )
            self.state = HSMMRuntimeState(
                filter_state=filter_state,
                observations=obs.clone(),
                duration_log_prob=duration,
                transition_log_prob=transition,
                tail_end_probability=_tail_end_probability(
                    self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                ),
                encoder_state=encoder_state,
                last_encoder_context=context.clone(),
                last_timestamp=timestamp if previous.uses_timestamps else None,
                uses_timestamps=previous.uses_timestamps,
            )
            self._uses_external_context = uses_external_context
            return filter_state

        if getattr(self.model, "encoder", None) is None:
            emission = _emission_log_prob(
                self.model,
                obs,
                None,
                self._score_cache,
            )
            filter_state = _filter_step_normalized(
                previous.filter_state,
                emission,
                previous.duration_log_prob,
                previous.transition_log_prob,
                previous.tail_end_probability,
            )
            duration, transition = _boundary_scores(
                self.model,
                None,
                temperature=self.temperature,
                cache=self._score_cache,
                batch_size=obs.shape[0],
            )
            self.state = HSMMRuntimeState(
                filter_state=filter_state,
                observations=obs.clone(),
                duration_log_prob=duration,
                transition_log_prob=transition,
                tail_end_probability=_tail_end_probability(
                    self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
                ),
                encoder_state=None,
                last_timestamp=timestamp if previous.uses_timestamps else None,
                uses_timestamps=previous.uses_timestamps,
            )
            self._uses_external_context = uses_external_context
            return filter_state

        observations = torch.cat((previous.observations, obs), dim=1)
        sequence = self.model._build_sequence_set(observations)
        expected_lengths = torch.full_like(sequence.lengths, observations.shape[1])
        if not torch.equal(sequence.lengths, expected_lengths):
            raise ValueError("runtime filtering requires one valid observation per batch item")

        filter_state = _filter_step_normalized(
            previous.filter_state,
            sequence.log_probs[:, -1],
            previous.duration_log_prob,
            previous.transition_log_prob,
            previous.tail_end_probability,
        )
        duration, transition = _boundary_scores(
            self.model,
            sequence.contexts[:, -1:],
            temperature=self.temperature,
            cache=self._score_cache,
        )
        self.state = HSMMRuntimeState(
            filter_state=filter_state,
            observations=sequence.sequences.clone(),
            duration_log_prob=duration,
            transition_log_prob=transition,
            tail_end_probability=_tail_end_probability(
                self.model, filter_state.log_posterior.shape[0], filter_state.log_posterior
            ),
            encoder_state=None,
            last_timestamp=timestamp if previous.uses_timestamps else None,
            uses_timestamps=previous.uses_timestamps,
        )
        self._uses_external_context = uses_external_context
        return filter_state
