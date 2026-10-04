from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from nhsmm.config import DTYPE, ModelConfig
from nhsmm.context import align_context_tensor
from nhsmm.distributions import Duration, Emission, Initial, Transition
from nhsmm.models.base import DistributionSet as BaseDistributionSet
from nhsmm.models.base import NHSMM as BaseNHSMM


class DistributionSet(BaseDistributionSet):
    """Canonical distribution container used by the training model."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        initial: type[nn.Module] = Initial,
        duration: type[nn.Module] = Duration,
        transition: type[nn.Module] = Transition,
        emission: type[nn.Module] = Emission,
    ):
        super().__init__(
            config=config,
            initial=initial,
            duration=duration,
            transition=transition,
            emission=emission,
        )


class NHSMM(BaseNHSMM):
    """Canonical NHSMM with robust training/restart/context-calibration semantics."""

    def initialize_encoder(self, encoder: Optional[nn.Module] = None) -> None:
        super().initialize_encoder(encoder=encoder)

    def _align_external_context(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        if context is None:
            return None
        X_tensor = self._ensure_tensor(X)
        B, T, _ = X_tensor.shape
        if isinstance(context, list):
            if len(context) != B:
                raise ValueError(
                    f"context list length {len(context)} does not match batch size {B}"
                )
            if isinstance(X, list):
                expected_lengths = [int(torch.as_tensor(item).shape[0]) for item in X]
            else:
                expected_lengths = [T] * B
            for index, (item, expected_length) in enumerate(zip(context, expected_lengths)):
                item_tensor = torch.as_tensor(item)
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
            context = self._ensure_tensor(context)
        elif not torch.is_tensor(context):
            raise TypeError(f"Unsupported context type: {type(context)}")
        return align_context_tensor(
            context,
            batch_size=B,
            timesteps=T,
            context_dim=int(self.context_dim),
            device=X_tensor.device,
            dtype=X_tensor.dtype,
        )

    def _build_sequence_set(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[torch.Tensor] = None,
    ):
        aligned = self._align_external_context(X, context)
        return super()._build_sequence_set(X, context=aligned)

    def log_likelihood(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reduce: bool = False,
    ) -> torch.Tensor:
        aligned = self._align_external_context(X, context)
        return super().log_likelihood(X, context=aligned, reduce=reduce)

    def predict(
        self,
        X: Union[torch.Tensor, List[torch.Tensor]],
        context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mode: str = "viterbi",
        verbose: bool = True,
    ):
        aligned = self._align_external_context(X, context)
        return super().predict(X, context=aligned, mode=mode, verbose=verbose)

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
        super()._initialize_run_state(
            run_idx,
            observations=observations,
            encoder_state=encoder_state,
            emission_init_mode=emission_init_mode,
            context=context,
        )

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
        context_tensor = self._align_external_context(X, context)
        super().optimize(X, context=context_tensor, cfg=cfg)

        if cfg.transition_refine_steps <= 0:
            return self
        if context_tensor is None:
            raise ValueError("transition_refine_steps > 0 requires explicit external context")

        # Base optimize already selected/restored the best joint initialization.
        X_tensor = self._ensure_tensor(X)
        self._refine_transition_likelihood(
            X_tensor,
            context_tensor,
            steps=cfg.transition_refine_steps,
            lr=cfg.transition_refine_lr,
        )
        return self
