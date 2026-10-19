from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
from typing import Optional, List, Tuple, Any, Literal, Dict, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils.rnn import pad_sequence

from nhsmm.models import forward as forward_impl
from nhsmm.models import decoding as decoding_impl
from nhsmm.models.objective import observation_log_likelihood
from nhsmm.training import optimize_model
from nhsmm.encoder import DefaultEncoder
from nhsmm.context import ContextEncoder, ContextRouter, SequenceSet, _align_sequence_context
from nhsmm.distributions import Initial, Duration, Transition, Emission
from nhsmm.config import DTYPE, logger, ModelConfig


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
            tail_enabled=self.config.duration_tail,
            tail_init_probability=self.config.duration_tail_init_probability,
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
            self.dist.to(
                device=self.duration_logits_bias.device, dtype=self.duration_logits_bias.dtype
            )
            self.dist.initialize(context, jitter)
        except Exception as err:
            raise RuntimeError(f"Failed to initialize NHSMM PDFs: {err}") from err

    def _require_distributions(self) -> DistributionSet:
        if self.dist is None:
            raise RuntimeError(
                "NHSMM distributions are not initialized; call initialize_distributions() first"
            )
        return self.dist

    def _require_generative_context(self, context) -> None:
        if context is None and self.encoder is not None and not self.config.causal:
            raise ValueError(
                "generative likelihood/training with an internal encoder requires causal=True; "
                "noncausal observation-derived context defines retrospective scores only"
            )

    def _validate_optimization_config(self, cfg: ModelConfig) -> None:
        structural_fields = (
            "n_states",
            "n_features",
            "max_duration",
            "duration_tail",
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
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if self.context_dim is None:
            raise ValueError("external context requires ModelConfig.context_dim to be configured")
        if mask is None:
            X, mask = self._ensure_tensor(X, return_mask=True)
        return _align_sequence_context(
            context,
            lengths=mask.sum(dim=1),
            timesteps=X.shape[1],
            context_dim=int(self.context_dim),
            prototype=X,
        )

    def _build_sequence_set(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> SequenceSet:
        self._require_distributions()
        X, mask = self._ensure_tensor(X, return_mask=True)
        context = self._align_external_context(X, context, mask=mask)
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
                if B == 0 or T == 0:
                    context_tensor = X.new_zeros((B, T, self.context_dim))
                    canonical = X.new_zeros((B, 1, self.context_dim))
                else:
                    context_tensor, canonical = self.encoder.encode(sequences=X, mask=mask)
                if self.config.causal:
                    canonical = context_tensor.new_zeros((B, 1, context_tensor.shape[-1]))
        else:
            context_tensor = context
            canonical = (
                context_tensor[:, :1] if T else X.new_zeros((B, 1, context_tensor.shape[-1]))
            )

        K = self.config.n_states
        emission_context = context_tensor if context_tensor.shape[-1] > 0 else None
        if context is None and self.encoder is not None and self.config.causal and T > 0:
            # Predict x_t from context available before observing x_t. Boundary
            # distributions retain unshifted contexts and use their previous row.
            emission_context = torch.cat((canonical, context_tensor[:, :-1]), dim=1)
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
    def _segment_duration_log_score(
        log_duration_at_end: torch.Tensor,
        duration: int,
        tail_end_probability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return completed-segment duration score for one actual duration.

        ``log_duration_at_end`` is ``[..., K, D]``. Without tail mode,
        ``duration`` must lie in ``1..D``. With tail mode, entries ``1..D-1``
        remain exact and any completed segment ``duration >= D`` uses the
        geometric tail mass ``q_tail * (1-h)^(duration-D) * h``.
        """
        D = log_duration_at_end.shape[-1]
        if duration < 1:
            raise ValueError("duration must be >= 1")
        if tail_end_probability is None:
            if duration > D:
                raise ValueError("finite-support duration exceeds max_duration")
            return log_duration_at_end[..., duration - 1]
        if duration < D:
            return log_duration_at_end[..., duration - 1]

        tail = tail_end_probability.to(
            device=log_duration_at_end.device, dtype=log_duration_at_end.dtype
        )
        while tail.ndim < log_duration_at_end.ndim - 1:
            tail = tail.unsqueeze(0)
        score = log_duration_at_end[..., -1] + tail.log()
        excess = duration - D
        if excess > 0:
            log_continue = torch.where(
                tail < 1.0,
                (1.0 - tail).clamp_min(torch.finfo(tail.dtype).tiny).log(),
                torch.full_like(tail, float("-inf")),
            )
            score = score + excess * log_continue
        return score

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
        return forward_impl._forward_causal_hazard(self, X, router, temperature)

    def forward(
        self,
        X: SequenceSet,
        context: Optional[Union[torch.Tensor, ContextRouter]] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        return forward_impl.forward(self, X, context, temperature, timestep)

    @staticmethod
    def _viterbi_terminal_index(score: torch.Tensor) -> int:
        return decoding_impl._viterbi_terminal_index(score)

    def _viterbi_causal_hazard(
        self,
        X: SequenceSet,
        router: ContextRouter,
    ) -> List[torch.Tensor]:
        return decoding_impl._viterbi_causal_hazard(self, X, router)

    def _viterbi_noncausal_tail(
        self,
        X: SequenceSet,
        router: ContextRouter,
    ) -> List[torch.Tensor]:
        return decoding_impl._viterbi_noncausal_tail(self, X, router)

    def _viterbi(
        self, X: SequenceSet, context: Optional[Union[torch.Tensor, ContextRouter]] = None
    ) -> List[torch.Tensor]:

        return decoding_impl._viterbi(self, X, context)

    def log_likelihood(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reduce: bool = False,
    ) -> torch.Tensor:
        self._require_distributions()
        self._require_generative_context(context)
        seq_set = self._build_sequence_set(X, context=context)
        log_likelihoods = observation_log_likelihood(self, seq_set)

        if torch.isposinf(log_likelihoods).any():
            raise ValueError("log_likelihood produced +inf")
        if torch.isnan(log_likelihoods).any():
            raise ValueError("log_likelihood produced NaN")
        return log_likelihoods.sum() if reduce else log_likelihoods

    def predict(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mode: Literal["viterbi", "log_likelihood"] = "viterbi",
        verbose: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        if mode == "log_likelihood":
            return self.log_likelihood(X, context=context, reduce=False)
        if mode != "viterbi":
            raise ValueError(f"Unsupported decoding mode '{mode}'")

        X_tensor = self._ensure_tensor(X)
        B, T, F = X_tensor.shape
        if B == 0 or T == 0:
            return [torch.empty(0, dtype=torch.long, device=X_tensor.device) for _ in range(B)]
        if verbose:
            logger.info(f"[Predict] Sequences: {B}, max_len: {T}")

        seq_set = self._build_sequence_set(X, context=context)
        router = ContextRouter.from_tensor(seq_set)
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
        dtype = self.duration_logits_bias.dtype

        if torch.is_tensor(X):
            if X.ndim == 2:
                X = X.unsqueeze(0)
            elif X.ndim != 3:
                raise ValueError(f"Unsupported X shape {X.shape}")
            X = X.to(device=device, dtype=dtype)
            if not torch.isfinite(X).all():
                raise ValueError("observations must contain only finite values")
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=device)
            return (X, mask) if return_mask else X

        if isinstance(X, list):
            if not X:
                if return_mask:
                    return torch.empty(
                        0, 0, self.config.n_features, device=device, dtype=dtype
                    ), torch.empty(0, 0, dtype=torch.bool, device=device)
                return torch.empty(0, 0, self.config.n_features, device=device, dtype=dtype)

            X_tensors = [torch.as_tensor(x, device=device, dtype=dtype) for x in X]
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
            rows = [
                torch.as_tensor(
                    item,
                    device=self.duration_logits_bias.device,
                    dtype=self.duration_logits_bias.dtype,
                )
                for item in X
            ]
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
    def _initialize_emission_from_observations(
        self, observations: torch.Tensor, *, cluster_variance: bool = False
    ) -> None:
        emission = self.dist.emission
        observations = observations.to(
            device=self.duration_logits_bias.device, dtype=self.duration_logits_bias.dtype
        )
        centers = self._kmeans_centers(observations, self.config.n_states)
        if emission.emission_type == "gaussian":
            emission.mu.copy_(centers)
            if cluster_variance:
                labels = torch.cdist(observations, centers).argmin(dim=-1)
                global_variance = observations.var(dim=0, unbiased=False)
                variances = torch.stack(
                    [
                        (
                            observations[labels == state].var(dim=0, unbiased=False)
                            if int((labels == state).sum()) > 1
                            else global_variance
                        )
                        for state in range(self.config.n_states)
                    ]
                )
                target_variance = variances.clamp(min=0.05, max=5.0)
                # Gaussian variance is softplus(log_var), not exp(log_var).
                inverse_softplus = target_variance + torch.log(-torch.expm1(-target_variance))
                emission.log_var.copy_(inverse_softplus)
            else:
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
        self.dist = self.dist.fresh().to(
            device=self.duration_logits_bias.device, dtype=self.duration_logits_bias.dtype
        )
        self.dist.train(self.training)

        # Base distribution parameters are initialized independently of
        # observed context. Context modulation is learned through likelihood.
        for name in ("initial", "transition", "duration"):
            getattr(self.dist, name).initialize(context=None)

        mode = emission_init_mode or self.config.emission_init_mode
        emission = self.dist.emission
        if mode in ("kmeans", "kmeans_cluster_variance"):
            if mode == "kmeans_cluster_variance" and emission.emission_type != "gaussian":
                raise ValueError("kmeans_cluster_variance requires gaussian emissions")
            self._initialize_emission_from_observations(
                observations, cluster_variance=(mode == "kmeans_cluster_variance")
            )
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

        self._require_generative_context(context)
        seq_set = self._build_sequence_set(X, context=context)
        temperature = t_min + (t_max - t_min) / (
            1 + math.exp((10 / max_iter) * (it - max_iter / 2))
        )
        log_likelihoods = observation_log_likelihood(self, seq_set, temperature)

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

    def _optimization_chunks(self, X, context, batch_size):
        """Slice whole independent sequences; preserve lengths and aligned context."""
        if batch_size is None:
            yield X, context
            return
        if torch.is_tensor(X) and X.ndim == 2:
            X = X.unsqueeze(0)
        for start in range(0, len(X), batch_size):
            chunk = X[start : start + batch_size]
            length = (
                max(item.shape[0] for item in chunk) if isinstance(chunk, list) else chunk.shape[1]
            )
            ctx = None if context is None else context[start : start + batch_size, :length]
            yield chunk, ctx

    def _backward_training_loss(self, X, context, batch_size, **kwargs):
        """Accumulate sum-likelihood gradients, applying the global penalty once."""
        total_ll = self.duration_logits_bias.new_zeros(())
        total_loss = total_ll.clone()
        for index, (chunk, ctx) in enumerate(self._optimization_chunks(X, context, batch_size)):
            options = {**kwargs, "loss_bias": kwargs["loss_bias"] if index == 0 else 0.0}
            ll, loss = self._compute_loss(chunk, context=ctx, **options)
            if loss.requires_grad:
                loss.backward()
            total_ll = total_ll + ll.detach()
            total_loss = total_loss + loss.detach()
        return total_ll, total_loss

    def _target_training_score(self, X, context, temperature, batch_size=None):
        """Deterministic fixed-temperature score, preserving the training mode."""
        training = self.training
        self.eval()
        try:
            with torch.no_grad():
                ll = self.duration_logits_bias.new_zeros(())
                for chunk, ctx in self._optimization_chunks(X, context, batch_size):
                    part, _ = self._compute_loss(
                        chunk,
                        context=ctx,
                        loss_bias=0.0,
                        it=0,
                        t_min=temperature,
                        t_max=temperature,
                    )
                    ll = ll + part
            return ll
        finally:
            self.train(training)

    def optimization_summary(self) -> dict:
        """Return independent telemetry from the last completed optimization.

        A new optimization attempt invalidates the previous summary before it
        changes training state. Partial/failed fits never publish a completed
        summary. This telemetry is not stored in inference artifacts.
        """
        summary = getattr(self, "_optimization_summary", None)
        if summary is None:
            raise RuntimeError("No completed optimization summary available")
        return deepcopy(summary)

    def fit(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cfg: Optional[ModelConfig] = None,
    ) -> "NHSMM":
        """Initialize if necessary, optimize whole sequences, and return this model.

        Uses the same optimizer/restarts as optimize(). Existing distribution
        factories are retained. This does not switch to eval/frozen inference,
        infer sequence boundaries, or accept independent-row sample weights.
        """
        if self.dist is None:
            self.initialize_distributions()
        return self.optimize(X, context=context, cfg=cfg)

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
        return optimize_model(self, X, context=context, cfg=cfg)
