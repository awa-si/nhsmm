from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from nhsmm.config import DTYPE, ModelConfig, logger
from nhsmm.distributions import Duration, Emission, Initial, Transition
from nhsmm.models.base import DistributionSet as BaseDistributionSet
from nhsmm.models.base import NHSMM as BaseNHSMM


class DistributionSet(BaseDistributionSet):
    """Training-aware distribution container with robust context capacity."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        initial: Optional[nn.Module] = Initial,
        duration: Optional[nn.Module] = Duration,
        transition: Optional[nn.Module] = Transition,
        emission: Optional[nn.Module] = Emission,
    ):
        nn.Module.__init__(self)
        self.config = config
        if self.config is None:
            raise ValueError("DistributionSet requires ModelConfig")

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


class NHSMM(BaseNHSMM):
    """Canonical NHSMM with robust training/restart/context-calibration semantics."""

    def initialize_distributions(
        self,
        context: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        dist: Optional[type[BaseDistributionSet]] = None,
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

    def _initialize_run_state(
        self,
        run_idx: int,
        observations: torch.Tensor,
        encoder_state: Optional[Dict[str, torch.Tensor]] = None,
        emission_init_mode: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize one independent optimization run."""
        dist_type = type(self.dist)
        self.dist = dist_type(config=self.config).to(device=self.device, dtype=DTYPE)

        # Base parameters are initialized independently of observed context.
        # Context modulation is learned through the likelihood objective.
        for name in ("initial", "transition", "duration"):
            getattr(self.dist, name).initialize(context=None)

        mode = emission_init_mode or self.config.emission_init_mode
        emission = self.dist.emission
        if mode == "kmeans":
            self._initialize_emission_from_observations(observations)
        else:
            if mode == "randome":
                mode = "random"
            emission.initialize(mode=mode, context=None)

        with torch.no_grad():
            self.duration_logits_bias.fill_(1.0)

        if self.encoder is not None:
            if encoder_state is not None:
                self.encoder.load_state_dict(encoder_state)
            self.encoder.reset()

        if hasattr(self, "_convergence"):
            self._convergence.converged_flags[run_idx] = False

    def _refine_transition_likelihood(
        self,
        X: torch.Tensor,
        context: Optional[torch.Tensor],
        *,
        steps: int,
        lr: float,
    ) -> float:
        """Refine transition context modulation against exact sequence likelihood."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if lr <= 0.0:
            raise ValueError("lr must be positive")

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
            for p in self.parameters():
                p.requires_grad_(id(p) in transition_ids)
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
            for p in self.parameters():
                p.requires_grad_(original_requires_grad[id(p)])
            self.train(original_training)

    def optimize(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cfg: Optional[ModelConfig] = None,
    ):
        """Optimize joint parameters and optionally calibrate transition context modulation."""
        cfg = cfg or self.config
        super().optimize(X, context=context, cfg=cfg)

        if cfg.transition_refine_steps <= 0:
            return self

        # Base optimize already selected/restored the best joint initialization.
        X_tensor = self._ensure_tensor(X)
        context_tensor = self._ensure_tensor(context)
        self._refine_transition_likelihood(
            X_tensor,
            context_tensor,
            steps=cfg.transition_refine_steps,
            lr=cfg.transition_refine_lr,
        )
        return self
