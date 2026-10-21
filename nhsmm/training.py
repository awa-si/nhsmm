"""Optimization orchestration for NHSMM; no inference or distribution definitions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from nhsmm.config import ModelConfig, logger
from nhsmm.convergence import Convergence
from nhsmm.context import BASE_CONTEXT, CausalContext

if TYPE_CHECKING:
    from nhsmm.models.base import NHSMM


def optimize_model(
    model: NHSMM,
    X: torch.Tensor | list[torch.Tensor],
    context: torch.Tensor | list[torch.Tensor] | None = None,
    cfg: ModelConfig | None = None,
) -> NHSMM:
    """Run independent restarts and restore the caller's model mode.

    The model owns initialization, objective evaluation and parameter snapshots.
    This function owns update ordering, optimizer/scheduler setup, convergence
    and final restart selection. Diagnostics stay on the model so existing
    consumers observe the same histories and optimizer state.
    """
    model._require_distributions()

    cfg = cfg or model.config
    cfg.__post_init__()
    model._require_generative_context(context)
    model._validate_optimization_config(cfg)
    if cfg.transition_refine_steps > 0 and (context is None or context is BASE_CONTEXT):
        raise ValueError("transition_refine_steps > 0 requires explicit external context")
    model._optimization_summary = None
    original_training = model.training
    model.train()
    try:
        observations = model._training_observations(X)
        if observations.ndim != 2 or observations.shape[-1] != model.config.n_features:
            raise ValueError("training observations must have the configured feature dimension")
        if observations.shape[0] == 0:
            raise ValueError("training requires at least one observation")
        if not torch.isfinite(observations).all():
            raise ValueError("training observations must contain only finite values")
        context = model._align_external_context(X, context)
        if (
            cfg.transition_refine_steps > 0
            and isinstance(context, CausalContext)
            and context.available is not None
            and not bool(context.available.any())
        ):
            raise ValueError("transition refinement requires available context rows")

        model._convergence = Convergence(
            tol=cfg.tol,
            rel_tol=cfg.tol,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
            mode=cfg.convergence_mode,
            plateau_tol=cfg.plateau_tol,
            plateau_window=cfg.plateau_window,
            patience=cfg.convergence_patience or max(1, cfg.plateau_window // 2),
            patience_max=cfg.convergence_patience_max,
            patience_scale=cfg.convergence_patience_scale,
            auto_patience=cfg.convergence_auto_patience,
            verbose=cfg.verbose,
        )

        encoder_state = (
            model._clone_state_dict(model.encoder) if model.encoder is not None else None
        )

        model._restart_scores = []
        model._optimization_history = []
        model._restart_score_temperature = cfg.optimizer_target_temperature
        best_score = -float("inf")
        completed_updates = [0] * cfg.n_init
        for run_idx in range(cfg.n_init):
            if cfg.optimizer_initialization == "restart":
                model._initialize_run_state(
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
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not name.endswith(".log_temperature")
            ]
            if not params:
                raise RuntimeError("NHSMM has no trainable parameters")

            model._optimizer = torch.optim.Adam(
                params, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps
            )
            if cfg.use_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    model._optimizer,
                    mode="max",
                    factor=cfg.scheduler_factor,
                    patience=max(cfg.scheduler_min_patience, cfg.plateau_window // 2),
                )
                model._convergence.attach_scheduler(scheduler)

            prev_ll = None
            history = []
            model._optimization_history.append(history)
            for it in range(cfg.max_iter):
                model._optimizer.zero_grad()
                target = cfg.optimizer_target_temperature
                minimum = (
                    target
                    if cfg.optimizer_temperature_mode == "constant"
                    else cfg.temperature_min_ratio * target
                )
                ll, loss = model._backward_training_loss(
                    X,
                    context=context,
                    batch_size=cfg.optimizer_batch_size,
                    loss_bias=cfg.loss_bias,
                    it=it,
                    max_iter=cfg.max_iter,
                    t_min=minimum,
                    t_max=target,
                    temperature_steepness=cfg.temperature_steepness,
                )
                target_ll = model._target_training_score(
                    X, context, target, cfg.optimizer_batch_size
                )
                history.append(
                    {
                        "completed_updates": it,
                        "scheduled_ll": float(ll.detach()),
                        "target_ll": float(target_ll.detach()),
                        "target_temperature": target,
                        "scheduled_temperature": minimum
                        + (target - minimum)
                        / (
                            1
                            + math.exp(
                                (cfg.temperature_steepness / cfg.max_iter) * (it - cfg.max_iter / 2)
                            )
                        ),
                    }
                )
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                model._optimizer.step()
                completed_updates[run_idx] += 1
                project_tail = getattr(model.dist.duration, "_project_tail_parameter", None)
                if callable(project_tail):
                    project_tail()

                ll_val = float(target_ll.item())
                converged = model._convergence.update(ll_val, it, run_idx)

                if cfg.verbose:
                    delta = ll_val - prev_ll if prev_ll is not None else float("nan")
                    logger.info(
                        f"[Iter {it:03d}] target LL={ll_val:.6f} Δ={delta:.3e}; scheduled LL={float(ll.detach()):.6f}"
                    )

                if converged and cfg.convergence_stop:
                    if cfg.verbose:
                        logger.info(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

                prev_ll = ll_val

            # Score the restart under the parameters that actually remain
            # after the final optimizer step. ll_val above is computed before
            # optimizer.step(), so it is stale with respect to the snapshot.
            final_ll = model._target_training_score(
                X, context, cfg.optimizer_target_temperature, cfg.optimizer_batch_size
            )
            final_ll_val = float(final_ll.item())
            model._restart_scores.append(final_ll_val)

            if final_ll_val > best_score:
                best_score = final_ll_val
                model._snapshot_best_params()

        if cfg.n_init > 1:
            model._restore_best_params()

        if cfg.transition_refine_steps > 0:
            model._refine_transition_likelihood(
                X,
                context,
                steps=cfg.transition_refine_steps,
                lr=cfg.transition_refine_lr,
                cfg=cfg,
            )

        model._optimization_summary = {
            "schema": "nhsmm-optimization-v1",
            "main_updates_per_run": completed_updates,
            "main_updates": sum(completed_updates),
            "transition_refinement_updates": cfg.transition_refine_steps,
            "selected_run": max(range(cfg.n_init), key=model._restart_scores.__getitem__),
            "restart_score_temperature": cfg.optimizer_target_temperature,
            "post_main_restart_scores": model._restart_scores.copy(),
        }
        return model
    finally:
        model.train(original_training)
