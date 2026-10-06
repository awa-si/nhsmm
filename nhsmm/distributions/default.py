from __future__ import annotations
from typing import Optional, Tuple, Dict, Type
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.distributions import Distribution, MultivariateNormal, StudentT

from nhsmm.config import EPS, MIN_LOGITS, MAX_LOGITS, NEG_INF


class Categorical(Distribution):
    has_rsample = True
    arg_constraints = {
        "logits": torch.distributions.constraints.real,
        "probs": torch.distributions.constraints.simplex,
    }

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        dim=-1,
        validate_args=False,
    ):
        super().__init__(validate_args=validate_args)
        if (logits is None) == (probs is None):
            raise ValueError("Specify exactly one of logits or probs.")

        reference = logits if logits is not None else probs
        if not isinstance(reference, torch.Tensor):
            raise TypeError("logits/probs must be torch.Tensor values")
        if not reference.is_floating_point():
            raise TypeError("logits/probs must use a floating dtype")
        if reference.ndim < 1 or reference.shape[-1] < 1:
            raise ValueError("Categorical requires a non-empty category dimension")
        normalized_dim = dim if dim >= 0 else reference.ndim + dim
        if normalized_dim != reference.ndim - 1:
            raise ValueError("Categorical only supports the last dimension as the category axis")
        self.dim = -1
        if logits is not None:
            if not torch.isfinite(logits).all():
                raise ValueError("logits must contain only finite values")
            self._logits = logits
            self._log_probs = nnF.log_softmax(logits, dim=dim)
            self._probs = self._log_probs.exp()
        else:
            if not torch.isfinite(probs).all() or (probs < 0).any():
                raise ValueError("probs must be finite and non-negative")
            total = probs.sum(dim=-1, keepdim=True)
            if (total <= 0).any():
                raise ValueError("probs must have positive mass on every batch row")
            self._probs = probs / total
            self._log_probs = self._probs.clamp_min(EPS).log()
            self._logits = self._log_probs

    @property
    def logits(self):
        return self._logits

    @property
    def probs(self):
        return self._probs

    @property
    def log_probs(self):
        return self._log_probs

    @property
    def mean(self):
        return self._probs

    @property
    def batch_shape(self):
        return self._logits.shape[:-1]

    @property
    def event_shape(self):
        return torch.Size()

    def sample(self, sample_shape=torch.Size()):
        return torch.distributions.Categorical(probs=self._probs).sample(sample_shape)

    def rsample(
        self, sample_shape=torch.Size(), temperature: Optional[float] = None, hard: bool = False
    ):
        tau = 1.0 if temperature is None else float(temperature)
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        logits_exp = self._logits
        if sample_shape:
            logits_exp = logits_exp.expand(*sample_shape, *self.batch_shape, self._logits.shape[-1])
        return nnF.gumbel_softmax(logits_exp, tau=tau, hard=hard, dim=-1)

    def log_prob(self, value: torch.Tensor):
        value = torch.as_tensor(value, device=self._log_probs.device)
        if value.is_floating_point():
            if not torch.isfinite(value).all():
                raise ValueError("categorical observations must contain only finite values")
            if not torch.equal(value, value.round()):
                raise ValueError("categorical observations must be integer-valued")
        value = value.long()
        n_states = self._log_probs.shape[-1]
        if ((value < 0) | (value >= n_states)).any():
            raise ValueError(f"categorical observations must be in [0, {n_states - 1}]")
        log_probs = self._log_probs
        while log_probs.ndim < value.ndim + 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = log_probs.expand(*value.shape, n_states)
        return log_probs.gather(-1, value.unsqueeze(-1)).squeeze(-1)

    def entropy(self):
        return -(self._probs * self._log_probs).sum(dim=-1)

    def mode(self):
        return self._logits.argmax(dim=self.dim)


class IndependentStudentT(Distribution):
    has_rsample = True
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "df": torch.distributions.constraints.positive,
        "scale": torch.distributions.constraints.positive,
    }

    def __init__(self, loc, scale, df, event_dim=1, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.df = df
        self.event_dim = event_dim
        shape = torch.broadcast_shapes(self.loc.shape, self.scale.shape, self.df.shape)
        if event_dim < 0 or event_dim > len(shape):
            raise ValueError("event_dim must be between 0 and parameter rank")
        batch_shape = torch.Size(shape[:-event_dim]) if event_dim else torch.Size(shape)
        event_shape = torch.Size(shape[-event_dim:]) if event_dim else torch.Size()
        super().__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )

    @property
    def batch_shape(self):
        shape = torch.broadcast_shapes(self.loc.shape, self.scale.shape, self.df.shape)
        if self.event_dim == 0:
            return torch.Size(shape)
        return torch.Size(shape[: -self.event_dim])

    @property
    def event_shape(self):
        if self.event_dim == 0:
            return torch.Size()
        shape = torch.broadcast_shapes(self.loc.shape, self.scale.shape, self.df.shape)
        return torch.Size(shape[-self.event_dim :])

    @property
    def mean(self):
        return torch.where(self.df > 1.0, self.loc, torch.full_like(self.loc, torch.nan))

    @property
    def variance(self):
        finite = self.scale.square() * self.df / (self.df - 2.0)
        return torch.where(
            self.df > 2.0,
            finite,
            torch.where(
                self.df > 1.0,
                torch.full_like(finite, torch.inf),
                torch.full_like(finite, torch.nan),
            ),
        )

    @property
    def stddev(self):
        return self.variance.sqrt()

    def rsample(self, sample_shape=torch.Size()):
        return StudentT(df=self.df, loc=self.loc, scale=self.scale).rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value):
        t = (value - self.loc) / self.scale
        df = self.df
        logp = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * torch.log(df * torch.pi)
            - torch.log(self.scale)
            - ((df + 1.0) / 2.0) * torch.log1p(t.square() / df)
        )
        if self.event_dim:
            dims = tuple(range(logp.ndim - self.event_dim, logp.ndim))
            logp = logp.sum(dim=dims)
        return logp

    def entropy(self):
        df = self.df
        entropy = (
            torch.log(self.scale)
            + 0.5 * torch.log(df)
            + torch.lgamma(df / 2.0)
            + torch.lgamma(torch.tensor(0.5, device=df.device, dtype=df.dtype))
            - torch.lgamma((df + 1.0) / 2.0)
            + ((df + 1.0) / 2.0) * (torch.digamma((df + 1.0) / 2.0) - torch.digamma(df / 2.0))
        )
        if self.event_dim:
            dims = tuple(range(entropy.ndim - self.event_dim, entropy.ndim))
            entropy = entropy.sum(dim=dims)
        return entropy

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        expanded_shape = batch_shape + self.event_shape
        return IndependentStudentT(
            loc=self.loc.expand(expanded_shape),
            scale=self.scale.expand(expanded_shape),
            df=self.df.expand(expanded_shape),
            event_dim=self.event_dim,
            validate_args=self._validate_args,
        )


class Neural(nn.Module, ABC):
    _dist_factory: Type[Distribution] = None

    def __init__(
        self,
        target_dim: int,
        activation: str = "tanh",
        final_activation: str = "tanh",
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        allow_projection: bool = True,
        delta_scale: float = 0.1,
        max_delta: float = 0.5,
    ):
        super().__init__()

        if self._dist_factory is None:
            raise TypeError(f"{type(self).__name__} must define _dist_factory")
        if not hasattr(self, "_shape"):
            raise TypeError(f"{type(self).__name__} must define _shape")
        if int(math.prod(self._shape)) != target_dim:
            raise ValueError("target_dim != prod(_shape)")

        self.max_delta = max_delta
        self.target_dim = target_dim
        self.context_dim = context_dim

        self.activation_fn = self._get_activation(activation)
        self.final_activation_fn = self._get_activation(final_activation)
        self._proj = (
            nn.Linear(context_dim, context_dim)
            if allow_projection and context_dim is not None
            else None
        )
        hidden_dim = hidden_dim or max(16, target_dim // 2, context_dim or target_dim)
        self.context_net = (
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Linear(hidden_dim, target_dim),
            )
            if context_dim is not None
            else None
        )

        self._init_weights(self.context_net)
        self._init_weights(self._proj)

        self.delta_scale = nn.Parameter(
            torch.tensor(delta_scale),
            requires_grad=context_dim is not None,
        )
        # Temperature is an explicit inference/training control, not an
        # independently learned model parameter. Keeping it frozen prevents a
        # generic optimizer over model.parameters() from changing HSMM
        # semantics behind NHSMM.optimize(), which already treats temperature
        # as scheduled/external.
        self.log_temperature = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.logits = nn.Parameter(torch.zeros(*self._shape))

    def _get_activation(self, mode: str = "tanh") -> nn.Module:
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "softplus": nn.Softplus,
            "identity": nn.Identity,
            "leaky_relu": lambda: nn.LeakyReLU(0.01),
        }
        key = mode.lower()
        try:
            factory = activations[key]
        except KeyError as exc:
            allowed = ", ".join(sorted(activations))
            raise ValueError(
                f"Unsupported activation {mode!r}; expected one of: {allowed}"
            ) from exc
        return factory()

    def _init_weights(self, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def base(self) -> torch.Tensor:
        return self.logits

    @base.setter
    def base(self, value: torch.Tensor) -> None:
        if value.shape != self._shape:
            raise ValueError(f"base shape {value.shape} does not match expected {self._shape}")
        self.logits.copy_(self._validate_base(value))

    def _dist_params(self, logits: torch.Tensor, **dist_kwargs) -> Dict[str, torch.Tensor]:
        return {"logits": logits, **dist_kwargs}

    def _get_dist(self, context=None, temperature=None, timestep=None, **dist_kwargs):
        mod = self._modulate(context, temperature, timestep)
        dist = self._dist_factory(**self._dist_params(mod, **dist_kwargs))
        return dist

    def _tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        k = len(self._shape)
        if tensor.ndim < k:
            raise ValueError(f"Tensor ndim={tensor.ndim} < required trailing dims={k}")
        if tensor.shape[-k:] != self._shape:
            tensor = tensor.reshape(*tensor.shape[:-k], *self._shape)
        return tensor

    def _apply_temperature(
        self, logits: torch.Tensor, temperature: Optional[float] = None
    ) -> torch.Tensor:
        tau = (
            self.log_temperature.exp()
            if temperature is None
            else torch.as_tensor(temperature, device=logits.device, dtype=logits.dtype)
        )
        if tau.numel() != 1 or not torch.isfinite(tau).all() or bool((tau <= 0).any()):
            raise ValueError("temperature must be a finite scalar > 0")
        return logits / tau

    def _prepare_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if context.ndim not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported context ndim={context.ndim}")
        if self.context_dim is not None and context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context_dim mismatch: got {context.shape[-1]}, expected {self.context_dim}"
            )
        if context.ndim == 1:
            return context.reshape(1, 1, -1)
        if context.ndim == 2:
            return context.unsqueeze(1)
        return context

    @staticmethod
    def _summarize_context(context: torch.Tensor) -> torch.Tensor:
        if context.ndim == 0:
            raise ValueError("context must have at least one dimension")
        if context.ndim == 1:
            return context
        return context.mean(dim=tuple(range(context.ndim - 1)))

    def _validate_base(self, base: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(base).all():
            bad_idx = (~torch.isfinite(base)).nonzero(as_tuple=True)
            raise ValueError(f"Non-finite base at {bad_idx}")
        return torch.clamp(base, MIN_LOGITS, MAX_LOGITS)

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if context is None or self.context_net is None:
            return torch.zeros_like(base)
        ctx = self._prepare_context(context)
        if timestep is not None:
            if ctx.ndim == 4:
                ctx = ctx[:, :, timestep : timestep + 1, :]
            elif ctx.ndim == 3:
                ctx = ctx[:, timestep : timestep + 1, :]
            elif ctx.ndim != 2:
                raise RuntimeError(f"Unsupported context ndim={ctx.ndim}")
        flat_ctx = ctx.reshape(-1, ctx.shape[-1])
        fused_weight = getattr(self, "_inference_context_weight", None)
        fused_bias = getattr(self, "_inference_context_bias", None)
        if (
            not self.training
            and fused_weight is not None
            and isinstance(self.context_net, nn.Sequential)
            and len(self.context_net) == 4
        ):
            delta = nnF.linear(flat_ctx, fused_weight, fused_bias)
            delta = self.context_net[1](delta)
            delta = self.context_net[2](delta)
            delta = self.context_net[3](delta)
        else:
            if self._proj is not None:
                flat_ctx = self._proj(flat_ctx)
            delta = self.context_net(flat_ctx)
        delta = delta.reshape(*ctx.shape[:-1], *self._shape)
        delta = self.final_activation_fn(delta)
        if grad_scale is not None:
            delta = delta * grad_scale
        delta = (delta * self.delta_scale).clamp(-self.max_delta, self.max_delta)
        if (
            timestep is not None
            and delta.ndim > len(self._shape)
            and delta.shape[-len(self._shape) - 1] == 1
        ):
            delta = delta.squeeze(-len(self._shape) - 1)
        return delta

    def _modulate(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        base = self._tensor_shape(self.base)
        delta = self._apply_context(base, context, timestep)
        mod = self._validate_base(base + delta)
        mod = self._apply_temperature(mod, temperature)
        return self._apply_constraints(mod, mask=kwargs.get("mask", None))

    @abstractmethod
    def _init_params(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def initialize(self, *args, **kwargs) -> Distribution:
        pass

    @abstractmethod
    def _apply_constraints(self, *args, **kwargs) -> torch.Tensor:
        pass

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        mod_logits = self._modulate(
            context=context, temperature=temperature, timestep=None, **kwargs
        )
        if return_dist:
            return self._dist_factory(**self._dist_params(mod_logits))
        return mod_logits

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.is_floating_point():
            if not torch.isfinite(x).all():
                raise ValueError("categorical observations must contain only finite values")
            if not torch.equal(x, x.round()):
                raise ValueError("categorical observations must be integer-valued")
        x = x.long()
        n_states = self._shape[-1]
        if ((x < 0) | (x >= n_states)).any():
            raise ValueError(f"categorical observations must be in [0, {n_states - 1}]")
        logits = self._modulate(
            context=context, temperature=temperature, timestep=timestep, **kwargs
        )
        if logits.shape[-1] != n_states:
            raise ValueError(f"Last dim of logits ({logits.shape[-1]}) != n_states ({n_states})")
        try:
            logits = logits.expand(*x.shape, n_states)
        except RuntimeError as e:
            raise ValueError(
                f"Cannot broadcast logits shape {logits.shape} to {x.shape + (n_states,)}"
            ) from e
        logp = nnF.log_softmax(logits, dim=-1)
        return logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)

    def expected_probs(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        return nnF.softmax(
            self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs),
            dim=-1,
        )


class Initial(Neural):
    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        activation: str,
        init_mode: str = "normal",
        allow_projection: bool = True,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
    ):
        self._shape = (n_states,)
        super().__init__(
            target_dim=n_states,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            allow_projection=allow_projection,
            final_activation="tanh",
            activation=activation,
        )
        self.n_states = n_states
        self.init_mode = init_mode
        with torch.no_grad():
            self.logits.copy_(self._init_params(mode=init_mode))

    def _init_params(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> torch.Tensor:
        mode = mode or self.init_mode
        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.n_states))
        elif mode == "biased":
            w = torch.linspace(0.8, 0.2, self.n_states)
            logits = torch.log(w / w.sum())
        elif mode == "normal":
            logits = torch.randn(self._shape) * 0.1
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        if context is not None and self.context_net is not None:
            ctx = self._summarize_context(context)
            logits = logits + self.context_net(ctx).view_as(logits)
        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter
        return logits

    @torch.no_grad
    def initialize(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> Distribution:
        self.logits.copy_(self._init_params(mode, None, jitter))
        return self._get_dist(context=context)

    def _apply_constraints(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return logits
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if mask.shape != (self.n_states,):
            raise ValueError("initial mask must have shape [K]")
        if not bool(mask.any()):
            raise ValueError("initial mask must allow at least one state")
        mask = mask.view((1,) * (logits.ndim - 1) + mask.shape)
        return logits.masked_fill(~mask, NEG_INF)

    def log_matrix(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        T: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        logits = self._modulate(
            context=context, temperature=temperature, timestep=timestep, **kwargs
        )
        if logits.ndim == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.ndim == 2:
            logits = logits.unsqueeze(0)
        return nnF.log_softmax(logits, dim=-1)


class Duration(Neural):
    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        activation: str,
        max_duration: int = 30,
        init_mode: str = "normal",
        allow_projection: bool = True,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
    ):
        if max_duration < 1:
            raise ValueError("max_duration must be >= 1")
        self._shape = (n_states, max_duration)
        super().__init__(
            target_dim=n_states * max_duration,
            allow_projection=allow_projection,
            context_dim=context_dim,
            final_activation="tanh",
            hidden_dim=hidden_dim,
            activation=activation,
        )
        self.n_states = n_states
        self.init_mode = init_mode
        self.max_duration = max_duration
        with torch.no_grad():
            self.logits.copy_(self._init_params(mode=init_mode))

    def _init_params(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> torch.Tensor:
        mode = mode or self.init_mode
        if mode == "uniform":
            logits = torch.full(self._shape, -math.log(self.max_duration))
        elif mode == "biased":
            w = torch.linspace(0.7, 0.3, self.max_duration).unsqueeze(0).expand(self.n_states, -1)
            logits = (w / w.sum(dim=1, keepdim=True)).clamp_min(EPS).log()
        elif mode == "normal":
            logits = torch.randn(*self._shape) * 0.1 - (
                torch.arange(self.max_duration) * 0.05
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        if context is not None and self.context_net is not None:
            ctx = self._summarize_context(context)
            ctx_delta = self.context_net(ctx).view_as(logits)
            logits = logits + ctx_delta
        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter
        return logits

    @torch.no_grad
    def initialize(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> Distribution:
        self.logits.copy_(self._init_params(mode, None, jitter))
        return self._get_dist(context=context)

    def _apply_constraints(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = logits.clamp(min=MIN_LOGITS, max=MAX_LOGITS)
        if mask is None:
            return logits
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if mask.ndim == 1:
            if mask.shape[0] != self.max_duration:
                raise ValueError("duration mask [D] must match max_duration")
            if not bool(mask.any()):
                raise ValueError("duration mask must allow at least one duration")
            mask = mask.view(*((1,) * (logits.ndim - 1)), self.max_duration)
        elif mask.ndim == 2:
            if mask.shape != (self.n_states, self.max_duration):
                raise ValueError("duration mask [K,D] must match n_states and max_duration")
            if not bool(mask.any(dim=-1).all()):
                raise ValueError("duration mask must allow at least one duration per state")
            mask = mask.view(*((1,) * (logits.ndim - 2)), self.n_states, self.max_duration)
        else:
            raise ValueError("duration mask must have shape [D] or [K,D]")
        return logits.masked_fill(~mask.expand_as(logits), NEG_INF)

    def log_matrix(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        T: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)
        soft_dmax = kwargs.get("soft_dmax", None)
        if soft_dmax is not None:
            gate = torch.sigmoid(soft_dmax).clamp_min(EPS)
            if gate.ndim == 1:
                if gate.shape[0] != self.max_duration:
                    raise ValueError("soft_dmax [D] must match max_duration")
                mod = mod + gate.log().view(1, 1, 1, -1)
            elif gate.ndim == 2:
                if gate.shape != (self.n_states, self.max_duration):
                    raise ValueError("soft_dmax [K,D] must match n_states and max_duration")
                mod = mod + gate.log().view(1, 1, *gate.shape)
            else:
                raise ValueError("soft_dmax must have shape [D] or [K, D]")
        logp = nnF.log_softmax(mod, dim=-1)
        while logp.ndim < 4:
            logp = logp.unsqueeze(0)
        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, *logp.shape[2:])
        return logp


class Transition(Neural):
    _dist_factory = Categorical

    def __init__(
        self,
        n_states: int,
        n_features: int,
        activation: str,
        transition_type: str,
        init_mode: str = "normal",
        allow_projection: bool = True,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        max_duration: Optional[int] = None,
    ):
        if max_duration is not None and max_duration < 1:
            raise ValueError("max_duration must be >= 1 when provided")
        self._shape = (
            (n_states, n_states) if max_duration is None else (n_states, max_duration, n_states)
        )
        self.target_dim = math.prod(self._shape)
        super().__init__(
            allow_projection=allow_projection,
            target_dim=self.target_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_activation="tanh",
            activation=activation,
        )
        self.n_states = n_states
        self.init_mode = init_mode
        self.max_duration = max_duration
        self.transition_type = transition_type
        self.register_buffer(
            "_structural_constraint",
            self._build_structural_constraint(),
            persistent=False,
        )
        with torch.no_grad():
            self.logits.copy_(self._init_params(mode=init_mode))

    def _build_structural_constraint(self) -> torch.Tensor:
        n, D = self.n_states, self.max_duration
        if self.transition_type == "ergodic":
            return torch.ones((n, n) if D is None else (n, D, n), dtype=torch.bool)
        if self.transition_type == "semi":
            if D is None:
                constraint = torch.eye(n, dtype=torch.bool)
                if n > 1:
                    idx = torch.arange(n - 1)
                    constraint[idx, idx + 1] = True
                return constraint
            constraint = torch.zeros(n, D, n, dtype=torch.bool)
            idx = torch.arange(n)
            constraint[idx, :, idx] = True
            if n > 1:
                idx = torch.arange(n - 1)
                constraint[idx, :, idx + 1] = True
            return constraint
        if self.transition_type == "left-to-right":
            if D is None:
                return torch.triu(torch.ones(n, n, dtype=torch.bool))
            constraint = torch.zeros(n, D, n, dtype=torch.bool)
            for offset in range(3):
                src = torch.arange(max(0, n - offset))
                constraint[src, :, src + offset] = True
            return constraint
        raise ValueError(f"Unsupported transition_type: {self.transition_type}")

    def _init_params(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> torch.Tensor:
        mode = mode or self.init_mode
        D, K, shape = self.max_duration, self.n_states, self._shape
        if mode == "uniform":
            logits = torch.full(shape, -math.log(K))
        elif mode == "biased":
            if D is None:
                m = torch.full((K, K), 0.1)
                m.fill_diagonal_(0.7)
                m /= m.sum(dim=-1, keepdim=True)
                logits = m.log()
            else:
                m = torch.full(shape, 0.1)
                for k in range(K):
                    m[k, :, k] = 0.7
                    m[k] /= m[k].sum(dim=-1, keepdim=True)
                logits = m.log()
        elif mode == "normal":
            logits = torch.randn(*shape) * 0.1
            if D is not None:
                logits = logits - (torch.arange(D) * 0.05).view(1, D, 1)
        else:
            raise ValueError(f"Unknown init_mode '{mode}'")
        if context is not None and self.context_net is not None:
            ctx = self._summarize_context(context)
            ctx_delta = self.context_net(ctx).view_as(logits)
            logits = logits + ctx_delta
        if jitter > 0.0:
            logits = logits + torch.randn_like(logits) * jitter
        return logits

    @torch.no_grad
    def initialize(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> Distribution:
        self.logits.copy_(self._init_params(mode, None, jitter))
        return self._get_dist(context=context)

    def _apply_constraints(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        constraint = self._structural_constraint.view(
            *((1,) * (logits.ndim - self._structural_constraint.ndim)),
            *self._structural_constraint.shape,
        ).expand_as(logits)
        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            expected = self._structural_constraint.shape
            if mask.shape != expected:
                raise ValueError(f"transition mask must have shape {tuple(expected)}")
            mask = mask.view(*((1,) * (logits.ndim - mask.ndim)), *mask.shape).expand_as(logits)
            constraint = constraint & mask
        if not bool(constraint.any(dim=-1).all()):
            raise ValueError("transition support must allow at least one destination per row")
        return logits.masked_fill(~constraint, NEG_INF)

    def log_matrix(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        timestep: Optional[int] = None,
        T: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        mod = self._modulate(context=context, temperature=temperature, timestep=timestep, **kwargs)
        logp = nnF.log_softmax(mod, dim=-1)
        expected_ndim = 4 if self.max_duration is None else 5
        while logp.ndim < expected_ndim:
            logp = logp.unsqueeze(0)
        if T is not None and logp.shape[1] == 1:
            logp = logp.expand(-1, T, *logp.shape[2:])
        return logp


class Emission(Neural):
    def __init__(
        self,
        n_states: int,
        n_features: int,
        activation: str,
        emission_type: str,
        min_covar: float = 1e-6,
        init_mode: str = "spread",
        allow_projection: bool = True,
        context_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        self._shape = (n_states, n_features)
        if emission_type == "gaussian":
            self._dist_factory = MultivariateNormal
        elif emission_type == "studentt":
            self._dist_factory = IndependentStudentT
        else:
            raise ValueError(f"Unsupported emission_type: {emission_type}")
        super().__init__(
            allow_projection=allow_projection,
            target_dim=n_states * n_features,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        # Emissions use mu/loc as their base parameters. The inherited logits
        # and temperature parameters remain in the state dict for artifact
        # compatibility but are not part of the emission likelihood.
        self.logits.requires_grad_(False)
        self.log_temperature.requires_grad_(False)
        self.n_states = n_states
        self.init_mode = init_mode
        self.min_covar = min_covar
        self.n_features = n_features
        self.emission_type = emission_type
        if self.emission_type == "gaussian":
            self.mu = nn.Parameter(torch.randn(self.n_states, self.n_features) * 0.1)
            self.log_var = nn.Parameter(torch.full((self.n_states, self.n_features), -1.0))
        else:
            self.dof = nn.Parameter(torch.full((n_states,), 5.0))
            self.loc = nn.Parameter(torch.randn(self.n_states, self.n_features) * 0.1)
            self.scale_param = nn.Parameter(torch.full((self.n_states, self.n_features), 0.1))

    @property
    def base(self):
        return self._validate_base(self.mu if self.emission_type == "gaussian" else self.loc)

    def _init_params(
        self,
        mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> torch.Tensor:
        mode = mode or self.init_mode
        reference = self.mu if self.emission_type == "gaussian" else self.loc
        device, dtype = reference.device, reference.dtype
        if mode == "random":
            init_mean = (
                torch.randn(self.n_states, self.n_features, device=device, dtype=dtype) * 0.1
            )
        elif mode == "spread":
            linspace = torch.linspace(-1.0, 1.0, steps=self.n_states, device=device, dtype=dtype)
            perm_idx = torch.stack(
                [torch.randperm(self.n_states, device=device) for _ in range(self.n_features)],
                dim=0,
            )
            init_mean = (
                linspace[perm_idx].T
                + torch.randn(self.n_states, self.n_features, device=device, dtype=dtype) * 0.05
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if context is not None and self.context_net is not None:
            ctx = self._summarize_context(context).to(device=device, dtype=dtype)
            delta = self.context_net(ctx).view(self.n_states, self.n_features)
            init_mean = init_mean + delta
            if self.emission_type == "studentt":
                with torch.no_grad():
                    self.dof.clamp_(min=2.1)
        init_var = torch.full(
            (self.n_states, self.n_features),
            1.0,
            device=init_mean.device,
            dtype=init_mean.dtype,
        )
        init_var += torch.rand_like(init_var) * jitter
        if self.emission_type == "gaussian":
            self.mu.copy_(init_mean)
            self.log_var.copy_(torch.log(init_var))
        else:
            self.loc.copy_(init_mean)
            self.scale_param.copy_(init_var.sqrt())
        return self.base

    @torch.no_grad()
    def initialize(
        self,
        mode: Optional[str] = "spread",
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
    ) -> Distribution:
        self._init_params(mode, None, jitter)
        return self._get_dist(context=context)

    def _apply_constraints(
        self, tensor: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if tensor is None or mask is None:
            return tensor
        mask = mask.to(dtype=torch.bool, device=tensor.device)
        while mask.ndim < tensor.ndim:
            mask = mask.unsqueeze(0)
        try:
            return tensor.masked_fill(~mask, 0.0)
        except RuntimeError as exc:
            raise ValueError(
                f"emission mask shape {tuple(mask.shape)} is not broadcastable to {tuple(tensor.shape)}"
            ) from exc

    def _apply_context(
        self,
        base: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        grad_scale: Optional[float] = None,
    ) -> torch.Tensor:
        delta = super()._apply_context(
            base, context=context, timestep=timestep, grad_scale=grad_scale
        )
        delta = delta - delta.mean(dim=-2, keepdim=True)
        max_abs = delta.abs().amax(dim=-2, keepdim=True)
        scale = torch.clamp(
            self.max_delta / max_abs.clamp_min(torch.finfo(delta.dtype).eps), max=1.0
        )
        return delta * scale

    def _apply_temperature(
        self, logits: torch.Tensor, temperature: Optional[float] = None
    ) -> torch.Tensor:
        return logits

    def _dist_params(self, loc: torch.Tensor, **dist_kwargs) -> dict:
        if loc.ndim == 2:
            loc = loc.unsqueeze(0).unsqueeze(0)
        B, T, K, n_feat = loc.shape
        if self.emission_type == "gaussian":
            var = nnF.softplus(self.log_var).clamp_min(self.min_covar)
            scale = torch.diag_embed(var.sqrt())[None, None, :, :, :].expand(
                B, T, K, n_feat, n_feat
            )
            return {"loc": loc, "scale_tril": scale, **dist_kwargs}
        scale = (
            nnF.softplus(self.scale_param)
            .clamp_min(self.min_covar)[None, None, :, :]
            .expand(B, T, K, n_feat)
        )
        df = (nnF.softplus(self.dof) + 2.0)[None, None, :, None].expand(B, T, K, 1)
        return {"loc": loc, "scale": scale, "df": df, **dist_kwargs}

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_dist: bool = False,
        **dist_kwargs,
    ) -> Tuple:
        loc = self._modulate(context=context, temperature=temperature, **dist_kwargs)
        if loc.ndim == 2:
            loc = loc.unsqueeze(0).unsqueeze(0)
        if return_dist:
            return self._dist_factory(**self._dist_params(loc))
        B, T, K, n_feat = loc.shape
        if self.emission_type == "gaussian":
            var = nnF.softplus(self.log_var).clamp_min(self.min_covar)
            covariance = torch.diag_embed(var)[None, None].expand(B, T, K, n_feat, n_feat)
            return loc, covariance
        scale = (
            nnF.softplus(self.scale_param)
            .clamp_min(self.min_covar)[None, None]
            .expand(B, T, K, n_feat)
        )
        return loc, scale

    def log_prob(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        **dist_kwargs,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, F = x.shape
        if F != self.n_features:
            raise ValueError(f"Feature mismatch: input F={F}, expected {self.n_features}")

        if self.emission_type == "gaussian":
            loc = self._modulate(
                context=context,
                temperature=temperature,
                **dist_kwargs,
            )
            if loc.ndim == 2:
                loc = loc.unsqueeze(0).unsqueeze(0)
            var = nnF.softplus(self.log_var).clamp_min(self.min_covar)
            diff = x[..., None, :] - loc
            logp = -0.5 * (diff.square() / var + var.log() + math.log(2.0 * math.pi)).sum(dim=-1)
        else:
            loc = self._modulate(
                context=context,
                temperature=temperature,
                **dist_kwargs,
            )
            if loc.ndim == 2:
                loc = loc.unsqueeze(0).unsqueeze(0)
            scale = nnF.softplus(self.scale_param).clamp_min(self.min_covar)[None, None]
            df = (nnF.softplus(self.dof) + 2.0)[None, None, :, None]
            t = (x[..., None, :] - loc) / scale
            logp = (
                torch.lgamma((df + 1.0) / 2.0)
                - torch.lgamma(df / 2.0)
                - 0.5 * torch.log(df * torch.pi)
                - torch.log(scale)
                - ((df + 1.0) / 2.0) * torch.log1p(t.square() / df)
            ).sum(dim=-1)

        if not torch.isfinite(logp).all():
            raise FloatingPointError("NaN/Inf in emission log-prob")
        return logp
