from __future__ import annotations
from dataclasses import replace
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils.rnn import pad_sequence

from nhsmm.convergence import Convergence
from nhsmm.encoder import DefaultEncoder
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet, align_context_tensor
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.filtering import duration_log_hazard
from nhsmm.config import DTYPE, logger, MIN_LOGITS, MAX_LOGITS, NEG_INF, ModelConfig


class DistributionSet(nn.Module):
    """Container for the four trainable HSMM distribution components."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        initial: type[nn.Module] = Initial,
        duration: type[nn.Module] = Duration,
        transition: type[nn.Module] = Transition,
        emission: type[nn.Module] = Emission,
    ):
        super().__init__()
        if config is None:
            raise ValueError("DistributionSet requires ModelConfig")
        self.config = config
        self._factories = {
            "initial": initial,
            "duration": duration,
            "transition": transition,
            "emission": emission,
        }
        distribution_hidden_dim = max(
            16,
            self.config.hidden_dim or 0,
            self.config.context_dim or 0,
        )
        self.initial = initial(
            hidden_dim=distribution_hidden_dim,
            context_dim=self.config.context_dim,
            n_states=self.config.n_states,
            init_mode=self.config.initial_init_mode,
            activation=self.config.activation,
        )
        self.duration = duration(
            hidden_dim=distribution_hidden_dim,
            context_dim=self.config.context_dim,
            n_states=self.config.n_states,
            max_duration=self.config.max_duration,
            init_mode=self.config.duration_init_mode,
            activation=self.config.activation,
        )
        self.transition = transition(
            hidden_dim=distribution_hidden_dim,
            context_dim=self.config.context_dim,
            n_states=self.config.n_states,
            n_features=self.config.n_features,
            transition_type=self.config.transition_type,
            init_mode=self.config.transition_init_mode,
            max_duration=self.config.max_duration,
            activation=self.config.activation,
        )
        self.transition.max_delta = self.config.transition_context_max_delta
        self.emission = emission(
            hidden_dim=distribution_hidden_dim,
            context_dim=self.config.context_dim,
            n_states=self.config.n_states,
            min_covar=self.config.min_covar,
            n_features=self.config.n_features,
            emission_type=self.config.emission_type,
            init_mode=self.config.emission_init_mode,
            activation=self.config.activation,
        )

    def initialize(
        self,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> Dict[str, Any]:
        return {
            "initial": self.initial.initialize(context=context, jitter=jitter),
            "duration": self.duration.initialize(context=context, jitter=jitter),
            "transition": self.transition.initialize(context=context, jitter=jitter),
            "emission": self.emission.initialize(context=context, jitter=jitter),
        }

    def fresh(self) -> "DistributionSet":
        """Recreate the same distribution container and injected component factories."""
        return type(self)(config=self.config, **self._factories)


class NHSMM(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        encoder: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        # Own a resolved configuration copy. Model construction may infer
        # encoder dimensions or mark an explicitly injected encoder as opt-in;
        # those resolutions must never mutate the caller's reusable config.
        self.config = replace(config)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        if encoder is not None:
            self.config.use_context_encoder = True

        self.context_dim = self.config.context_dim
        self.hidden_dim = self.config.hidden_dim

        self.dist: Optional[DistributionSet] = None
        self.duration_logits_bias = nn.Parameter(
            torch.ones(self.config.n_states, self.config.max_duration)
        )

        self.initialize_encoder(encoder=encoder)
        self.to(device=self.device, dtype=DTYPE)

    def initialize_encoder(self, encoder: Optional[nn.Module] = None) -> None:
        if encoder is None and not self.config.use_context_encoder:
            self.encoder = None
            return

        if encoder is None:
            if self.config.context_dim is None:
                encoder_hidden_dim = max(32, min(64, self.config.n_features * 2))
            else:
                directions = 1 if self.config.causal else 2
                if self.config.context_dim % directions != 0:
                    raise ValueError(
                        "context_dim must be divisible by 2 for the default "
                        "bidirectional encoder when causal=False"
                    )
                encoder_hidden_dim = self.config.context_dim // directions
            encoder = DefaultEncoder(
                n_features=self.config.n_features,
                cnn_channels=self.config.cnn_channels,
                cnn_kernel=self.config.cnn_kernel,
                hidden_dim=encoder_hidden_dim,
                bidirectional=not self.config.causal,
                causal=self.config.causal,
                dropout=self.config.dropout,
            )
        elif self.config.causal:
            raw_encoder = encoder.encoder if isinstance(encoder, ContextEncoder) else encoder
            if not bool(getattr(raw_encoder, "causal", False)):
                raise ValueError(
                    "ModelConfig.causal=True requires an encoder that explicitly declares causal=True."
                )

        self.encoder = (
            encoder
            if isinstance(encoder, ContextEncoder)
            else ContextEncoder(
                encoder=encoder,
                pool=self.config.pool,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
            )
        )
        self.encoder = self.encoder.to(device=self.device, dtype=DTYPE)

        try:
            self.encoder.eval()
            dummy = torch.zeros(
                1,
                16,
                self.config.n_features,
                device=self.device,
                dtype=DTYPE,
            )
            try:
                _, ctx, _ = self.encoder(
                    dummy,
                    return_context=True,
                    return_sequence=True,
                )
                inferred_dim = ctx.shape[-1]
            except TypeError:
                inferred_dim = self.encoder(dummy).shape[-1]

            if self.context_dim is None:
                self.context_dim = inferred_dim
            elif inferred_dim != self.context_dim:
                raise ValueError(
                    f"encoder output dimension ({inferred_dim}) must equal context_dim "
                    f"({self.context_dim})."
                )
            if self.hidden_dim is None:
                self.hidden_dim = self.context_dim
            elif self.hidden_dim != self.context_dim:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must equal context_dim "
                    f"({self.context_dim}) unless projections are explicitly defined."
                )
        finally:
            self.encoder.train()

        self.config.context_dim = self.context_dim
        self.config.hidden_dim = self.hidden_dim

    def initialize_distributions(
        self,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        dist: Optional[type[DistributionSet]] = None,
    ) -> None:
        if dist is not None:
            self.dist = dist(config=self.config)

        elif self.dist is None:
            self.dist = DistributionSet(config=self.config)

        try:
            self.dist.to(device=self.device, dtype=DTYPE)
            self.dist.initialize(context, jitter)
        except Exception as err:
            raise RuntimeError(f"Failed to initialize NHSMM PDFs: {err}") from err

    def _require_distributions(self) -> DistributionSet:
        if self.dist is None:
            raise RuntimeError(
                "NHSMM distributions are not initialized; call initialize_distributions() first"
            )
        return self.dist

    def _validate_optimization_config(self, cfg: ModelConfig) -> None:
        structural_fields = (
            "n_states",
            "n_features",
            "max_duration",
            "causal",
            "context_dim",
            "hidden_dim",
            "transition_type",
            "emission_type",
            "activation",
            "use_context_encoder",
        )
        mismatches = [
            name for name in structural_fields if getattr(cfg, name) != getattr(self.config, name)
        ]
        if mismatches:
            details = ", ".join(
                f"{name}={getattr(cfg, name)!r} (model {getattr(self.config, name)!r})"
                for name in mismatches
            )
            raise ValueError(
                "optimization config is structurally incompatible with this NHSMM: " + details
            )

    def _align_external_context(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if self.context_dim is None:
            raise ValueError("external context requires ModelConfig.context_dim to be configured")

        X_tensor = self._ensure_tensor(X)
        B, T, _ = X_tensor.shape
        context_dim = int(self.context_dim)
        if isinstance(context, list):
            if len(context) != B:
                raise ValueError(
                    f"context list length {len(context)} does not match batch size {B}"
                )
            expected_lengths = (
                [int(torch.as_tensor(item).shape[0]) for item in X]
                if isinstance(X, list)
                else [T] * B
            )
            context_tensors = []
            for index, (item, expected_length) in enumerate(zip(context, expected_lengths)):
                item_tensor = torch.as_tensor(
                    item,
                    device=X_tensor.device,
                    dtype=X_tensor.dtype,
                )
                if item_tensor.ndim != 2:
                    raise ValueError(
                        "context list entries must be [T,H] tensors; "
                        f"entry {index} has shape {tuple(item_tensor.shape)}"
                    )
                if item_tensor.shape[0] != expected_length:
                    raise ValueError(
                        f"context sequence {index} length {item_tensor.shape[0]} "
                        f"does not match observation length {expected_length}"
                    )
                if item_tensor.shape[1] != context_dim:
                    raise ValueError(
                        f"context sequence {index} feature dimension {item_tensor.shape[1]} "
                        f"does not match expected context_dim {context_dim}"
                    )
                if not torch.isfinite(item_tensor).all():
                    raise ValueError(f"context sequence {index} must contain only finite values")
                context_tensors.append(item_tensor)
            context = (
                pad_sequence(context_tensors, batch_first=True, padding_value=0.0)
                if context_tensors
                else X_tensor.new_empty((0, T, context_dim))
            )
        elif not torch.is_tensor(context):
            raise TypeError(f"Unsupported context type: {type(context)}")

        return align_context_tensor(
            context,
            batch_size=B,
            timesteps=T,
            context_dim=context_dim,
            device=X_tensor.device,
            dtype=X_tensor.dtype,
        )

    def _build_sequence_set(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> SequenceSet:
        self._require_distributions()
        context = self._align_external_context(X, context)
        X, mask = self._ensure_tensor(X, return_mask=True)
        B, T, F = X.shape
        if F != self.config.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.config.n_features}, got {F}"
            )

        if context is None:
            if self.encoder is None:
                context_tensor = X.new_zeros((B, T, 0))
                canonical = X.new_zeros((B, 1, 0))
            else:
                context_tensor, canonical = self.encoder.encode(sequences=X, mask=mask)
                if self.config.causal:
                    canonical = context_tensor[:, :1]
        else:
            context_tensor = context
            canonical = context_tensor[:, :1]

        K = self.config.n_states
        emission_context = context_tensor if context_tensor.shape[-1] > 0 else None
        if T == 0:
            log_probs = X.new_empty(B, 0, K)
        else:
            log_probs = self.dist.emission.log_prob(X, context=emission_context)
            log_probs = log_probs.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        return SequenceSet(
            sequences=X,
            lengths=mask.sum(dim=1),
            masks=mask.unsqueeze(-1),
            contexts=context_tensor,
            canonical=canonical,
            log_probs=log_probs,
        )

    @staticmethod
    def _active_context(context: torch.Tensor) -> Optional[torch.Tensor]:
        return context if context.shape[-1] > 0 else None

    @staticmethod
    def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(f"{name} batch dimension must be 1 or {batch_size}, got {tensor.shape[0]}")

    def _forward_causal_hazard(
        self,
        X: SequenceSet,
        router: ContextRouter,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Unnormalized causal forward recursion over ``(state, episode_age)``.

        Boundary ``t-1 -> t`` uses duration and transition quantities conditioned
        only on ``F_{t-1}``; ``x_t`` enters after propagation through its emission.
        """

        B, T, K = router.log_probs.shape[:3]
        D = int(self.dist.duration.max_duration)
        alpha = router.log_probs.new_full((B, T, K, D), float("-inf"))
        if T == 0:
            return alpha

        initial_logits = self.dist.initial.log_matrix(
            context=self._active_context(router.canonical),
            temperature=temperature,
            T=T,
        )
        duration_logits = self.dist.duration.log_matrix(
            context=self._active_context(router.context),
            temperature=temperature,
            T=T,
            soft_dmax=self.duration_logits_bias,
        )
        transition_logits = self.dist.transition.log_matrix(
            context=self._active_context(router.context),
            temperature=temperature,
            T=T,
            soft_dmax=self.duration_logits_bias,
        )

        initial_logits = self._expand_batch(initial_logits, B, "initial logits")
        duration_logits = self._expand_batch(duration_logits, B, "duration logits")
        transition_logits = self._expand_batch(transition_logits, B, "transition logits")

        expected_initial = (B, 1, K)
        expected_duration = (B, T, K, D)
        if initial_logits.shape != expected_initial:
            raise ValueError(
                f"initial logits must be {expected_initial}, got {initial_logits.shape}"
            )
        if duration_logits.shape != expected_duration:
            raise ValueError(
                f"duration logits must be {expected_duration}, got {duration_logits.shape}"
            )

        duration_dependent_transition = self.dist.transition.max_duration is not None
        expected_transition = (B, T, K, D, K) if duration_dependent_transition else (B, T, K, K)
        if transition_logits.shape != expected_transition:
            raise ValueError(
                f"transition logits must be {expected_transition}, got {transition_logits.shape}"
            )

        log_end, log_continue = duration_log_hazard(duration_logits.reshape(B * T, K, D))
        log_end = log_end.reshape(B, T, K, D)
        log_continue = log_continue.reshape(B, T, K, D)

        valid0 = X.lengths > 0
        alpha[valid0, 0, :, 0] = initial_logits[valid0, 0] + router.log_probs[valid0, 0]

        for t in range(1, T):
            valid = t < X.lengths
            if not bool(valid.any()):
                break
            active = torch.nonzero(valid, as_tuple=False).squeeze(-1)

            previous = alpha[active, t - 1]
            predicted = previous.new_full((active.numel(), K, D), float("-inf"))

            if D > 1:
                predicted[..., 1:] = previous[..., :-1] + log_continue[active, t - 1, :, :-1]

            boundary = previous + log_end[active, t - 1]
            if duration_dependent_transition:
                new_episode = torch.logsumexp(
                    boundary.unsqueeze(-1) + transition_logits[active, t - 1],
                    dim=(1, 2),
                )
            else:
                boundary_state = torch.logsumexp(boundary, dim=-1)
                new_episode = torch.logsumexp(
                    boundary_state.unsqueeze(-1) + transition_logits[active, t - 1],
                    dim=1,
                )
            predicted[..., 0] = new_episode
            alpha[active, t] = predicted + router.log_probs[active, t].unsqueeze(-1)

        return alpha

    def forward(
        self,
        X: SequenceSet,
        context: Optional[Union[torch.Tensor, ContextRouter]] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        self._require_distributions()

        router = (
            ContextRouter.from_tensor(X, context=context)
            if not isinstance(context, ContextRouter)
            else context
        )
        if self.config.causal:
            if timestep is not None:
                raise ValueError("timestep is not supported by the full causal forward recursion")
            return self._forward_causal_hazard(X, router, temperature=temperature)

        Dmax = self.dist.duration.max_duration
        B, T, K = router.log_probs.shape[:3]
        device = router.log_probs.device

        kwargs = dict(
            soft_dmax=self.duration_logits_bias,
            temperature=temperature,
            timestep=timestep,
            T=T,
        )
        initial_logits = self.dist.initial.log_matrix(
            context=self._active_context(router.canonical), **kwargs
        )
        duration_logits = self.dist.duration.log_matrix(
            context=self._active_context(router.context), **kwargs
        )
        transition_logits = self.dist.transition.log_matrix(
            context=self._active_context(router.context), **kwargs
        )
        initial_logits = self._expand_batch(initial_logits, B, "initial logits")
        duration_logits = self._expand_batch(duration_logits, B, "duration logits")
        transition_logits = self._expand_batch(transition_logits, B, "transition logits")

        cumsum_emit = torch.zeros((B, T + 1, K), device=device)
        cumsum_emit[:, 1:] = torch.cumsum(router.log_probs, dim=1)

        d_range = torch.arange(1, Dmax + 1, device=device)
        t_range = torch.arange(T, device=device).unsqueeze(1)
        start_idx = (t_range - d_range + 1).clamp(min=0)
        end_idx = t_range + 1

        start_idx = start_idx.unsqueeze(0).unsqueeze(2).expand(B, T, K, Dmax)
        end_idx = end_idx.unsqueeze(0).unsqueeze(2).expand(B, T, K, Dmax)

        cumsum_expand = cumsum_emit.unsqueeze(-1).expand(B, T + 1, K, Dmax)
        emit_sums = cumsum_expand.gather(1, end_idx) - cumsum_expand.gather(1, start_idx)
        emit_sums = emit_sums.clamp(min=MIN_LOGITS, max=MAX_LOGITS)

        alpha = torch.full((B, T, K, Dmax), NEG_INF, device=device)
        valid0 = X.lengths > 0
        alpha[valid0, 0, :, 0] = (
            initial_logits[valid0, 0]
            + duration_logits[valid0, 0, :, 0]
            + emit_sums[valid0, 0, :, 0]
        )

        for t in range(1, T):
            valid = t < X.lengths
            if not bool(valid.any()):
                break
            active = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            max_d = min(Dmax, t + 1)
            alpha_t = alpha.new_full((active.numel(), K, max_d), NEG_INF)

            for duration_index in range(max_d):
                duration = duration_index + 1
                start = t - duration + 1

                if start == 0:
                    predecessor = initial_logits[active, 0]
                else:
                    prev_t = start - 1
                    prev_alpha = alpha[active, prev_t]
                    if self.dist.transition.max_duration is None:
                        prev_state = torch.logsumexp(prev_alpha, dim=-1)
                        trans = transition_logits[active, prev_t]
                        predecessor = torch.logsumexp(
                            prev_state.unsqueeze(-1) + trans,
                            dim=1,
                        )
                    else:
                        trans = transition_logits[active, prev_t]
                        predecessor = torch.logsumexp(
                            prev_alpha.unsqueeze(-1) + trans,
                            dim=(1, 2),
                        )

                alpha_t[..., duration_index] = (
                    predecessor
                    + duration_logits[active, t, :, duration_index]
                    + emit_sums[active, t, :, duration_index]
                )

            full_alpha = alpha.new_full((active.numel(), K, Dmax), NEG_INF)
            full_alpha[..., :max_d] = alpha_t
            alpha[active, t] = full_alpha

        length_mask = torch.arange(T, device=device).unsqueeze(0) < X.lengths.unsqueeze(1)
        alpha = alpha.masked_fill(~length_mask.unsqueeze(-1).unsqueeze(-1), NEG_INF)
        return alpha

    def _viterbi_causal_hazard(
        self,
        X: SequenceSet,
        router: ContextRouter,
    ) -> List[torch.Tensor]:
        """MAP causal decoding over ``(state, episode_age)`` under dynamic hazards."""

        K = self.config.n_states
        D = int(self.dist.duration.max_duration)
        predicted: List[torch.Tensor] = []

        for b in range(router.log_probs.shape[0]):
            L = int(X.lengths[b].item())
            if L == 0:
                predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
                continue

            initial_logits = self.dist.initial.log_matrix(
                context=self._active_context(router.canonical[b : b + 1]), T=L
            )[0, 0]
            duration_logits = self.dist.duration.log_matrix(
                context=self._active_context(router.context[b : b + 1, :L]),
                T=L,
                soft_dmax=self.duration_logits_bias,
            )[0]
            transition_logits = self.dist.transition.log_matrix(
                context=self._active_context(router.context[b : b + 1, :L]),
                T=L,
                soft_dmax=self.duration_logits_bias,
            )[0]
            log_end, log_continue = duration_log_hazard(duration_logits)

            score = router.log_probs.new_full((L, K, D), float("-inf"))
            prev_state = torch.full((L, K, D), -1, dtype=torch.long, device=score.device)
            prev_age = torch.full((L, K, D), -1, dtype=torch.long, device=score.device)
            score[0, :, 0] = initial_logits + router.log_probs[b, 0]

            state_indices = torch.arange(K, device=score.device).view(K, 1)
            age_indices = torch.arange(max(D - 1, 0), device=score.device).view(1, -1)
            duration_dependent_transition = self.dist.transition.max_duration is not None

            for t in range(1, L):
                if D > 1:
                    continuation = score[t - 1, :, :-1] + log_continue[t - 1, :, :-1]
                    score[t, :, 1:] = continuation + router.log_probs[b, t].unsqueeze(-1)
                    prev_state[t, :, 1:] = state_indices.expand(K, D - 1)
                    prev_age[t, :, 1:] = age_indices.expand(K, D - 1)

                boundary = score[t - 1] + log_end[t - 1]
                if duration_dependent_transition:
                    candidates = boundary.unsqueeze(-1) + transition_logits[t - 1]
                    flat = candidates.reshape(K * D, K)
                    boundary_score, flat_index = flat.max(dim=0)
                    boundary_prev_state = flat_index // D
                    boundary_prev_age = flat_index % D
                else:
                    boundary_state, boundary_age = boundary.max(dim=-1)
                    candidates = boundary_state.unsqueeze(-1) + transition_logits[t - 1]
                    boundary_score, boundary_prev_state = candidates.max(dim=0)
                    boundary_prev_age = boundary_age[boundary_prev_state]

                score[t, :, 0] = boundary_score + router.log_probs[b, t]
                prev_state[t, :, 0] = boundary_prev_state
                prev_age[t, :, 0] = boundary_prev_age

            flat_index = int(score[L - 1].reshape(-1).argmax().item())
            state = flat_index // D
            age = flat_index % D
            path = torch.empty(L, dtype=torch.long, device=score.device)

            for t in range(L - 1, -1, -1):
                path[t] = state
                if t == 0:
                    break
                state_next = int(prev_state[t, state, age].item())
                age_next = int(prev_age[t, state, age].item())
                if state_next < 0 or age_next < 0:
                    raise RuntimeError("causal Viterbi backpointer is incomplete")
                state, age = state_next, age_next

            predicted.append(path)

        return predicted

    def _viterbi(
        self, X: SequenceSet, context: Optional[Union[torch.Tensor, ContextRouter]] = None
    ) -> List[torch.Tensor]:

        K = self.config.n_states
        Dmax = self.dist.duration.max_duration

        router = (
            context
            if isinstance(context, ContextRouter)
            else ContextRouter.from_tensor(X, context=context)
        )
        if self.config.causal:
            return self._viterbi_causal_hazard(X, router)

        B, T_max, _ = router.log_probs.shape
        device = router.log_probs.device

        predicted: List[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, device=device)
        state_indices = torch.arange(K, device=device)

        for b in range(B):
            L = int(router.mask[b].sum())
            if L == 0:
                predicted.append(router.log_probs.new_empty(0, dtype=torch.long))
                continue

            initial_logits = self.dist.initial.log_matrix(
                context=self._active_context(router.canonical[b : b + 1]), T=L
            )[0, 0]
            duration_logits = self.dist.duration.log_matrix(
                context=self._active_context(router.context[b : b + 1, :L]),
                T=L,
                soft_dmax=self.duration_logits_bias,
            )[0]
            transition_logits = self.dist.transition.log_matrix(
                context=self._active_context(router.context[b : b + 1, :L]),
                T=L,
                soft_dmax=self.duration_logits_bias,
            )[0]

            emit_log = router.log_probs[b, :L]
            cumsum_emit = torch.zeros((L + 1, K), device=device)
            cumsum_emit[1:] = torch.cumsum(emit_log, dim=0)

            V = torch.full((L, K), NEG_INF, device=device)
            back_ptr = torch.full((L, K), -1, dtype=torch.long, device=device)
            best_dur = torch.zeros((L, K), dtype=torch.long, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts.clamp_min(0)]).T
                scores_dur = duration_logits[t, :, :max_d] + emit_sums

                best_score_t = torch.full((K,), NEG_INF, device=device)
                best_prev_t = torch.full((K,), -1, dtype=torch.long, device=device)
                best_duration_t = torch.ones((K,), dtype=torch.long, device=device)

                for duration_index in range(max_d):
                    duration = duration_index + 1
                    start = t - duration + 1
                    segment_score = scores_dur[:, duration_index]

                    if start == 0:
                        candidate = initial_logits + segment_score
                        candidate_prev = torch.full((K,), -1, dtype=torch.long, device=device)
                    else:
                        prev_t = start - 1
                        if self.dist.transition.max_duration is None:
                            trans = transition_logits[prev_t]
                        else:
                            prev_duration_index = (best_dur[prev_t] - 1).clamp(min=0)
                            trans = transition_logits[
                                prev_t,
                                state_indices,
                                prev_duration_index,
                                :,
                            ]

                        predecessor_scores = V[prev_t].unsqueeze(-1) + trans
                        predecessor, predecessor_state = predecessor_scores.max(dim=0)
                        candidate = predecessor + segment_score
                        candidate_prev = predecessor_state

                    improve = candidate > best_score_t
                    best_score_t = torch.where(improve, candidate, best_score_t)
                    best_prev_t = torch.where(improve, candidate_prev, best_prev_t)
                    best_duration_t = torch.where(
                        improve,
                        torch.full_like(best_duration_t, duration),
                        best_duration_t,
                    )

                V[t] = best_score_t
                back_ptr[t] = best_prev_t
                best_dur[t] = best_duration_t

            t = L - 1
            state = int(V[t].argmax())
            segments = []
            while t >= 0:
                d = int(best_dur[t, state])
                start = max(0, t - d + 1)
                segments.append((start, t, state))
                prev = int(back_ptr[t, state])
                t = start - 1
                if prev >= 0:
                    state = prev

            segments.reverse()
            path = torch.cat(
                [
                    router.log_probs.new_full((end - start + 1,), st, dtype=torch.long)
                    for start, end, st in segments
                ]
            )

            predicted.append(path[:L])
        return predicted

    def log_likelihood(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reduce: bool = False,
    ) -> torch.Tensor:
        self._require_distributions()
        X_tensor, mask = self._ensure_tensor(X, return_mask=True)
        B = X_tensor.shape[0]
        if X_tensor.shape[1] == 0:
            result = X_tensor.new_full((B,), NEG_INF)
            return result.sum() if reduce else result

        seq_set = self._build_sequence_set(X, context=context)
        alpha = self.forward(seq_set)

        lengths = seq_set.lengths
        log_likelihoods = alpha.new_full((B,), NEG_INF)
        valid = lengths > 0
        if valid.any():
            last_alpha = alpha[valid, lengths[valid] - 1]
            log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

        if torch.isposinf(log_likelihoods).any():
            raise ValueError("log_likelihood produced +inf")
        log_likelihoods = torch.nan_to_num(log_likelihoods, nan=NEG_INF, neginf=NEG_INF)
        return log_likelihoods.sum() if reduce else log_likelihoods

    def predict(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mode: Literal["viterbi", "log_likelihood"] = "viterbi",
        verbose: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:

        X_tensor = self._ensure_tensor(X)
        B, T, F = X_tensor.shape

        if B == 0 or T == 0:
            return [torch.empty(0, dtype=torch.long, device=X_tensor.device) for _ in range(B)]

        if verbose:
            logger.info(f"[Predict] Sequences: {B}, max_len: {T}")

        seq_set = self._build_sequence_set(X, context=context)
        router = ContextRouter.from_tensor(seq_set)

        if mode == "viterbi":
            results = [torch.empty(0, dtype=torch.long, device=X_tensor.device) for _ in range(B)]
            nonzero_idx = torch.nonzero(seq_set.lengths, as_tuple=False).squeeze(-1)

            if len(nonzero_idx) > 0:
                seq_set_nz = seq_set.select(nonzero_idx)
                router_nz = router.select(nonzero_idx)
                decoded_paths = self._viterbi(seq_set_nz, context=router_nz)

                for i, path in zip(nonzero_idx.tolist(), decoded_paths):
                    length = int(seq_set.lengths[i].item())
                    results[i] = path[:length].to(dtype=torch.long)
            return results

        if mode == "log_likelihood":
            return self.log_likelihood(X, context=context, reduce=False)

        raise ValueError(f"Unsupported decoding mode '{mode}'")

    def decode(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mode: Literal["viterbi"] = "viterbi",
        first_only: bool = True,
        verbose: bool = True,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:

        X_tensor = self._ensure_tensor(X)
        B = X_tensor.shape[0]
        if verbose:
            logger.info(f"[decode] mode={mode}, batch_size={B}")

        preds = self.predict(X, mode=mode, context=context, verbose=verbose)
        return preds[0] if first_only and B == 1 else preds

    def _ensure_tensor(
        self, X: Union[torch.Tensor, List[torch.Tensor], None], return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor], None]:

        if X is None:
            return (None, None) if return_mask else None

        device = self.duration_logits_bias.device

        if torch.is_tensor(X):
            if X.ndim == 2:
                X = X.unsqueeze(0)
            elif X.ndim != 3:
                raise ValueError(f"Unsupported X shape {X.shape}")
            X = X.to(device=device, dtype=DTYPE)
            if not torch.isfinite(X).all():
                raise ValueError("observations must contain only finite values")
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=device)
            return (X, mask) if return_mask else X

        if isinstance(X, list):
            if not X:
                if return_mask:
                    return torch.empty(0, 0, 0, device=device), torch.empty(
                        0, 0, dtype=torch.bool, device=device
                    )
                return torch.empty(0, 0, 0, device=device)

            X_tensors = [torch.as_tensor(x, device=device, dtype=DTYPE) for x in X]
            for index, item in enumerate(X_tensors):
                if item.ndim != 2:
                    raise ValueError(
                        "observation list entries must be [T,F] tensors; "
                        f"entry {index} has shape {tuple(item.shape)}"
                    )
                if item.shape[1] != self.config.n_features:
                    raise ValueError(
                        f"observation sequence {index} feature dimension {item.shape[1]} "
                        f"does not match expected {self.config.n_features}"
                    )
                if not torch.isfinite(item).all():
                    raise ValueError(
                        f"observation sequence {index} must contain only finite values"
                    )
            lengths = [x.shape[0] for x in X_tensors]
            X_padded = pad_sequence(
                X_tensors, batch_first=True, padding_value=self.config.pad_value
            )
            mask = torch.zeros(X_padded.shape[:2], dtype=torch.bool, device=device)
            for i, L in enumerate(lengths):
                mask[i, :L] = 1
            return (X_padded, mask) if return_mask else X_padded

        raise TypeError(f"Unsupported type: {type(X)}")

    def _training_observations(self, X: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if torch.is_tensor(X):
            tensor = self._ensure_tensor(X)
            return tensor.reshape(-1, tensor.shape[-1])
        if isinstance(X, list):
            if not X:
                raise ValueError("X must contain at least one sequence")
            rows = [torch.as_tensor(item, device=self.device, dtype=DTYPE) for item in X]
            return torch.cat(rows, dim=0)
        raise TypeError(f"Unsupported type: {type(X)}")

    @staticmethod
    def _kmeans_centers(
        observations: torch.Tensor, n_states: int, n_iter: int = 20
    ) -> torch.Tensor:
        """Return K-Means++/Lloyd centers for training-time emission initialization."""
        if observations.ndim != 2:
            raise ValueError("observations must be a [N,F] tensor")
        if observations.shape[0] < n_states:
            raise ValueError("observations must contain at least n_states rows")
        if not torch.isfinite(observations).all():
            raise ValueError("observations must be finite")

        x = observations.detach()
        n = x.shape[0]
        first = int(torch.randint(n, (1,), device=x.device).item())
        centers = [x[first]]
        closest = (x - centers[0]).square().sum(dim=-1)
        for _ in range(1, n_states):
            total = closest.sum()
            if not torch.isfinite(total) or float(total) <= 0.0:
                index = int(torch.randint(n, (1,), device=x.device).item())
            else:
                index = int(torch.multinomial(closest / total, 1).item())
            centers.append(x[index])
            closest = torch.minimum(closest, (x - centers[-1]).square().sum(dim=-1))

        current = torch.stack(centers)
        for _ in range(n_iter):
            distances = torch.cdist(x, current).square()
            labels = distances.argmin(dim=-1)
            nearest = distances.min(dim=-1).values
            updated = []
            for state in range(n_states):
                members = labels == state
                updated.append(
                    x[members].mean(dim=0) if bool(members.any()) else x[nearest.argmax()]
                )
            candidate = torch.stack(updated)
            if torch.allclose(candidate, current, atol=1e-5, rtol=1e-5):
                return candidate
            current = candidate
        return current

    @torch.no_grad()
    def _initialize_emission_from_observations(self, observations: torch.Tensor) -> None:
        emission = self.dist.emission
        observations = observations.to(device=self.device, dtype=DTYPE)
        centers = self._kmeans_centers(observations, self.config.n_states)
        if emission.emission_type == "gaussian":
            emission.mu.copy_(centers)
            emission.log_var.zero_()
        else:
            emission.loc.copy_(centers)
            emission.scale_param.fill_(1.0)
            emission.dof.clamp_(min=2.1)

    def _initialize_run_state(
        self,
        run_idx: int,
        observations: torch.Tensor,
        encoder_state: Optional[Dict[str, torch.Tensor]] = None,
        emission_init_mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize one optimization run without warm-starting from another run."""
        self.dist = self.dist.fresh().to(device=self.device, dtype=DTYPE)
        self.dist.train(self.training)

        # Base distribution parameters are initialized independently of
        # observed context. Context modulation is learned through likelihood.
        for name in ("initial", "transition", "duration"):
            getattr(self.dist, name).initialize(context=None)

        mode = emission_init_mode or self.config.emission_init_mode
        emission = self.dist.emission
        if mode == "kmeans":
            self._initialize_emission_from_observations(observations)
        else:
            emission.initialize(mode=mode, context=None)

        with torch.no_grad():
            self.duration_logits_bias.fill_(1.0)

        if self.encoder is not None:
            if encoder_state is not None:
                self.encoder.load_state_dict(encoder_state)
            self.encoder.reset()

        if hasattr(self, "_convergence"):
            self._convergence.converged_flags[run_idx] = False

    @staticmethod
    def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in module.state_dict().items()}

    def _snapshot_best_params(self):
        """Save an independent snapshot of the best optimization run."""
        self._best_state = {
            "dist": self._clone_state_dict(self.dist),
            "duration_logits_bias": self.duration_logits_bias.detach().clone(),
        }
        if self.encoder is not None:
            self._best_state["encoder"] = self._clone_state_dict(self.encoder)

    def _restore_best_params(self):
        """Restore parameters from the best independent optimization run."""
        if not hasattr(self, "_best_state"):
            raise RuntimeError("No best parameters snapshot available.")

        self.dist.load_state_dict(self._best_state["dist"])
        with torch.no_grad():
            self.duration_logits_bias.copy_(self._best_state["duration_logits_bias"])
        if self.encoder is not None and "encoder" in self._best_state:
            self.encoder.load_state_dict(self._best_state["encoder"])

    def _compute_loss(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        loss_bias: float = 1e-3,
        it: int = 0,
        max_iter: int = 20,
        t_min: float = 0.3,
        t_max: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        seq_set = self._build_sequence_set(X, context=context)
        temperature = t_min + (t_max - t_min) / (
            1 + math.exp((10 / max_iter) * (it - max_iter / 2))
        )
        alpha = self.forward(seq_set, temperature=temperature)
        lengths = seq_set.lengths

        log_likelihoods = alpha.new_full((len(seq_set.sequences),), NEG_INF)
        valid = lengths > 0
        if valid.any():
            last_alpha = alpha[valid, lengths[valid] - 1]
            log_likelihoods[valid] = torch.logsumexp(last_alpha.flatten(1), dim=1)

        ll = log_likelihoods.sum()

        loss = -ll
        if self.duration_logits_bias.shape[1] > 1 and loss_bias != 0.0:
            loss = loss + (
                loss_bias
                * nnF.relu(
                    self.duration_logits_bias[:, 1:] - self.duration_logits_bias[:, :-1]
                ).mean()
            )

        return ll, loss

    def _refine_transition_likelihood(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        *,
        steps: int,
        lr: float,
    ) -> float:
        """Refine transition context modulation against exact sequence likelihood."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if context is None:
            raise ValueError("transition refinement requires explicit external context")

        transition = self.dist.transition
        transition_params = [
            *[p for p in transition.context_net.parameters() if p.requires_grad],
            transition.delta_scale,
        ]
        transition_params = [p for p in transition_params if p.requires_grad]
        if not transition_params:
            raise RuntimeError("transition context modulation has no trainable parameters")

        original_requires_grad = {id(p): p.requires_grad for p in self.parameters()}
        transition_ids = {id(p) for p in transition_params}
        original_training = self.training
        try:
            for parameter in self.parameters():
                parameter.requires_grad_(id(parameter) in transition_ids)
            self.eval()
            optimizer = torch.optim.Adam(transition_params, lr=lr)
            for step in range(steps):
                optimizer.zero_grad()
                _, loss = self._compute_loss(
                    X,
                    context=context,
                    loss_bias=0.0,
                    it=step,
                    max_iter=steps,
                    t_min=1.0,
                    t_max=1.0,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transition_params, 5.0)
                optimizer.step()

            with torch.no_grad():
                ll, _ = self._compute_loss(
                    X,
                    context=context,
                    loss_bias=0.0,
                    it=steps,
                    max_iter=steps,
                    t_min=1.0,
                    t_max=1.0,
                )
            return float(ll.item())
        finally:
            for parameter in self.parameters():
                parameter.requires_grad_(original_requires_grad[id(parameter)])
            self.train(original_training)

    def optimize(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cfg: Optional[ModelConfig] = None,
    ):
        """
        Optimize NHSMM parameters using the provided configuration.

        Args:
            X: Input sequences (tensor or list of tensors)
            context: Optional context features
            cfg: ModelConfig object controlling learning and convergence
        """
        self._require_distributions()

        cfg = cfg or self.config
        self._validate_optimization_config(cfg)
        original_training = self.training
        self.train()
        try:
            observations = self._training_observations(X)
            context = self._align_external_context(X, context)

            self._convergence = Convergence(
                tol=cfg.tol,
                rel_tol=cfg.tol,
                n_init=cfg.n_init,
                max_iter=cfg.max_iter,
                mode=cfg.convergence_mode,
                plateau_tol=cfg.plateau_tol,
                early_stop=cfg.convergence_stop,
                plateau_window=cfg.plateau_window,
                patience=max(1, cfg.plateau_window // 2),
                verbose=cfg.verbose,
            )

            encoder_state = (
                self._clone_state_dict(self.encoder) if self.encoder is not None else None
            )

            self._restart_scores = []
            best_score = -float("inf")
            for run_idx in range(cfg.n_init):
                self._initialize_run_state(
                    run_idx,
                    observations=observations,
                    encoder_state=encoder_state,
                    emission_init_mode=cfg.emission_init_mode,
                    context=context,
                )

                if cfg.verbose:
                    logger.info(f"\n=== Run {run_idx + 1}/{cfg.n_init} ===")

                # Optimize every trainable model parameter. The causal context
                # encoder participates in the likelihood graph and must be updated
                # alongside the probabilistic components.
                params = [
                    parameter
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad and not name.endswith(".log_temperature")
                ]
                if not params:
                    raise RuntimeError("NHSMM has no trainable parameters")

                self._optimizer = torch.optim.Adam(params, lr=cfg.lr)
                if cfg.use_scheduler:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self._optimizer,
                        mode="max",
                        factor=0.5,
                        patience=max(2, cfg.plateau_window // 2),
                    )
                    self._convergence.attach_scheduler(scheduler)

                prev_ll = None
                for it in range(cfg.max_iter):
                    self._optimizer.zero_grad()
                    ll, loss = self._compute_loss(
                        X, context=context, loss_bias=cfg.loss_bias, it=it, max_iter=cfg.max_iter
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    self._optimizer.step()

                    ll_val = float(ll.item())
                    converged = self._convergence.update(ll_val, it, run_idx)

                    if cfg.verbose:
                        delta = ll_val - prev_ll if prev_ll is not None else float("nan")
                        logger.info(f"[Iter {it:03d}] LL={ll_val:.6f} Δ={delta:.3e}")

                    if converged:
                        if cfg.verbose:
                            logger.info(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                        break

                    prev_ll = ll_val

                # Score the restart under the parameters that actually remain
                # after the final optimizer step. ll_val above is computed before
                # optimizer.step(), so it is stale with respect to the snapshot.
                with torch.no_grad():
                    final_ll, _ = self._compute_loss(
                        X,
                        context=context,
                        loss_bias=cfg.loss_bias,
                        it=it,
                        max_iter=cfg.max_iter,
                    )
                final_ll_val = float(final_ll.item())
                self._restart_scores.append(final_ll_val)

                if final_ll_val > best_score:
                    best_score = final_ll_val
                    self._snapshot_best_params()

            if cfg.n_init > 1:
                self._restore_best_params()

            if cfg.transition_refine_steps > 0:
                if context is None:
                    raise ValueError(
                        "transition_refine_steps > 0 requires explicit external context"
                    )
                self._refine_transition_likelihood(
                    X,
                    context,
                    steps=cfg.transition_refine_steps,
                    lr=cfg.transition_refine_lr,
                )

            return self
        finally:
            self.train(original_training)
