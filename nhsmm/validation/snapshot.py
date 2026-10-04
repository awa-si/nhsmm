from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import torch

from nhsmm.diagnostics import ModelHealthReport, ModelHealthThresholds, evaluate_model_health

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidationSnapshot:
    """Reproducible, domain-neutral evidence for one fitted model on one dataset."""

    schema_version: int
    label: Optional[str]
    model_fingerprint: str
    data_fingerprint: str
    context_fingerprint: Optional[str]
    n_sequences: int
    n_timesteps: int
    n_features: int
    sequence_lengths: tuple[int, ...]
    log_likelihood_per_sequence: tuple[float, ...]
    log_likelihood_total: float
    log_likelihood_per_timestep: float
    state_switch_rate: float
    mean_run_length: float
    health: ModelHealthReport

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "model_fingerprint": self.model_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "n_sequences": self.n_sequences,
            "n_timesteps": self.n_timesteps,
            "n_features": self.n_features,
            "sequence_lengths": list(self.sequence_lengths),
            "log_likelihood_per_sequence": list(self.log_likelihood_per_sequence),
            "log_likelihood_total": self.log_likelihood_total,
            "log_likelihood_per_timestep": self.log_likelihood_per_timestep,
            "state_switch_rate": self.state_switch_rate,
            "mean_run_length": self.mean_run_length,
            "health": self.health.as_dict(),
        }


@dataclass(frozen=True)
class ValidationComparison:
    """Difference between two validation snapshots."""

    reference_label: Optional[str]
    candidate_label: Optional[str]
    same_model: bool
    log_likelihood_per_timestep_delta: float
    effective_states_delta: float
    max_occupancy_delta: float
    posterior_entropy_delta: float
    state_switch_rate_delta: float
    mean_run_length_delta: float
    occupancy_l1_distance: float
    reference_healthy: bool
    candidate_healthy: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _update_tensor_hash(hasher: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode())
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    hasher.update(value.numpy().tobytes(order="C"))


def model_fingerprint(model: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for config + model state."""

    config = getattr(model, "config", None)
    state_dict = getattr(model, "state_dict", None)
    if config is None or not callable(state_dict):
        raise TypeError("model must expose config and state_dict()")

    try:
        config_data = asdict(config)
    except TypeError as exc:
        raise TypeError("model.config must be a dataclass instance") from exc

    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(config_data, sort_keys=True, separators=(",", ":"), default=str).encode()
    )
    for name, tensor in sorted(state_dict().items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("model state_dict must map string names to tensors")
        hasher.update(name.encode())
        _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def _masked_tensor_fingerprint(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> str:
    if tensor.ndim < 2 or mask.ndim != 2 or tensor.shape[:2] != mask.shape:
        raise ValueError("tensor and mask must agree on [B,T]")
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode())
    hasher.update(str(tensor.shape[-1]).encode())
    lengths = mask.sum(dim=1).to(dtype=torch.long)
    _update_tensor_hash(hasher, lengths)
    for index, length in enumerate(lengths.tolist()):
        _update_tensor_hash(hasher, tensor[index, : int(length)])
    return hasher.hexdigest()


def _decoded_path_statistics(decoded: list[torch.Tensor]) -> tuple[float, float]:
    total_steps = sum(int(path.numel()) for path in decoded)
    if total_steps == 0:
        raise ValueError("validation snapshot requires at least one decoded timestep")

    switches = 0
    transition_opportunities = 0
    runs = 0
    for path in decoded:
        length = int(path.numel())
        if length == 0:
            continue
        runs += 1
        if length > 1:
            changes = path[1:] != path[:-1]
            switches += int(changes.sum().item())
            transition_opportunities += length - 1
            runs += int(changes.sum().item())

    switch_rate = (
        float(switches / transition_opportunities) if transition_opportunities > 0 else 0.0
    )
    mean_run_length = float(total_steps / runs)
    return switch_rate, mean_run_length


def evaluate_validation_snapshot(
    model: Any,
    X: Union[torch.Tensor, list[torch.Tensor]],
    context: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    *,
    label: Optional[str] = None,
    health_thresholds: Optional[ModelHealthThresholds] = None,
) -> ValidationSnapshot:
    """Evaluate a fitted model on a fixed dataset without mutating model state."""

    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("label must be a non-empty string or None")
    if getattr(model, "dist", None) is None:
        raise RuntimeError("model distributions are not initialized")

    original_training = bool(model.training)
    try:
        model.eval()
        with torch.inference_mode():
            input_tensor, input_mask = model._ensure_tensor(X, return_mask=True)
            if input_mask is None or int(input_mask.sum().item()) == 0:
                raise ValueError("validation snapshot requires at least one valid timestep")
            if not torch.isfinite(input_tensor[input_mask]).all():
                raise ValueError("validation observations must contain only finite values")

            sequence = model._build_sequence_set(X, context=context)
            lengths = tuple(int(value) for value in sequence.lengths.tolist())
            n_timesteps = sum(lengths)
            if n_timesteps == 0:
                raise ValueError("validation snapshot requires at least one valid timestep")

            ll = model.log_likelihood(X, context=context, reduce=False)
            if ll.ndim != 1 or ll.numel() != len(lengths):
                raise RuntimeError("log_likelihood must return one value per sequence")
            if not torch.isfinite(ll).all():
                raise ValueError("validation log likelihood contains non-finite values")

            decoded_raw = model.predict(X, context=context, mode="viterbi", verbose=False)
            decoded = decoded_raw if isinstance(decoded_raw, list) else [decoded_raw]
            if tuple(int(path.numel()) for path in decoded) != lengths:
                raise RuntimeError(
                    "Viterbi output lengths do not match validation sequence lengths"
                )
            switch_rate, mean_run_length = _decoded_path_statistics(decoded)

            valid_mask = sequence.masks.squeeze(-1)
            data_hash = _masked_tensor_fingerprint(sequence.sequences, valid_mask)
            context_hash = (
                None
                if context is None
                else _masked_tensor_fingerprint(sequence.contexts, valid_mask)
            )

            health = evaluate_model_health(
                model,
                X,
                context=context,
                thresholds=health_thresholds,
            )

            total_ll = float(ll.sum().item())
            per_timestep = total_ll / n_timesteps
            return ValidationSnapshot(
                schema_version=SNAPSHOT_SCHEMA_VERSION,
                label=label,
                model_fingerprint=model_fingerprint(model),
                data_fingerprint=data_hash,
                context_fingerprint=context_hash,
                n_sequences=len(lengths),
                n_timesteps=n_timesteps,
                n_features=int(sequence.sequences.shape[-1]),
                sequence_lengths=lengths,
                log_likelihood_per_sequence=tuple(float(value) for value in ll.tolist()),
                log_likelihood_total=total_ll,
                log_likelihood_per_timestep=per_timestep,
                state_switch_rate=switch_rate,
                mean_run_length=mean_run_length,
                health=health,
            )
    finally:
        model.train(original_training)


def compare_validation_snapshots(
    reference: ValidationSnapshot,
    candidate: ValidationSnapshot,
    *,
    require_same_model: bool = True,
) -> ValidationComparison:
    """Compare two snapshots, typically train/reference versus held-out/OOS."""

    same_model = reference.model_fingerprint == candidate.model_fingerprint
    if require_same_model and not same_model:
        raise ValueError(
            "validation snapshots use different model fingerprints; "
            "state-index comparisons require the same fitted model"
        )
    if reference.health.occupancy.shape != candidate.health.occupancy.shape:
        raise ValueError("validation snapshots use incompatible state dimensions")

    occupancy_l1 = float(
        (reference.health.occupancy - candidate.health.occupancy).abs().sum().item()
    )
    return ValidationComparison(
        reference_label=reference.label,
        candidate_label=candidate.label,
        same_model=same_model,
        log_likelihood_per_timestep_delta=(
            candidate.log_likelihood_per_timestep - reference.log_likelihood_per_timestep
        ),
        effective_states_delta=(
            candidate.health.effective_states - reference.health.effective_states
        ),
        max_occupancy_delta=(candidate.health.max_occupancy - reference.health.max_occupancy),
        posterior_entropy_delta=(
            candidate.health.posterior_entropy_mean - reference.health.posterior_entropy_mean
        ),
        state_switch_rate_delta=(candidate.state_switch_rate - reference.state_switch_rate),
        mean_run_length_delta=(candidate.mean_run_length - reference.mean_run_length),
        occupancy_l1_distance=occupancy_l1,
        reference_healthy=reference.health.healthy,
        candidate_healthy=candidate.health.healthy,
    )
