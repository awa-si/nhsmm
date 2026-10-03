# nhsmm/context.py

from __future__ import annotations
from typing import Optional, Literal, Tuple, Callable, Dict, Sequence, Union
from dataclasses import dataclass, field
import contextlib
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils.rnn import pad_sequence


def align_context_tensor(
    context: torch.Tensor,
    *,
    batch_size: int,
    timesteps: int,
    context_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Normalize public external-context shapes to ``[B,T,H]``.

    Accepted shapes are ``[H]`` (global/static), ``[T,H]`` (time-varying and
    shared across the batch), ``[B,1,H]`` (batch-specific/static), and
    ``[B,T,H]`` (batch- and time-specific). ``[B,H]`` is intentionally not a
    public form because it is ambiguous with ``[T,H]`` whenever ``B == T``.
    """
    tensor = torch.as_tensor(context, device=device, dtype=dtype)
    B, T, H = batch_size, timesteps, context_dim

    if tensor.ndim == 1:
        if tensor.shape[0] != H:
            raise ValueError(f"1D context must be [H]=[{H}], got {tuple(tensor.shape)}")
        return tensor.view(1, 1, H).expand(B, T, H)

    if tensor.ndim == 2:
        if tensor.shape != (T, H):
            raise ValueError(
                f"2D context must be [T,H]=({T},{H}); "
                "batch-specific static context must use [B,1,H]"
            )
        return tensor.unsqueeze(0).expand(B, T, H)

    if tensor.ndim == 3:
        if tensor.shape == (B, 1, H):
            return tensor.expand(B, T, H)
        if tensor.shape == (B, T, H):
            return tensor
        raise ValueError(
            f"3D context must be [B,1,H] or [B,T,H] for "
            f"(B,T,H)=({B},{T},{H}), got {tuple(tensor.shape)}"
        )

    raise ValueError(
        f"Unsupported context shape {tuple(tensor.shape)}; expected [H], [T,H], [B,1,H], or [B,T,H]"
    )


@dataclass
class SequenceSet:
    """Container for batched sequences with optional contexts and log probabilities."""

    sequences: torch.Tensor
    lengths: torch.Tensor
    masks: torch.Tensor
    contexts: torch.Tensor
    canonical: torch.Tensor
    log_probs: Optional[torch.Tensor] = None

    @classmethod
    def from_unbatched(
        cls,
        sequences: Sequence[torch.Tensor],
        contexts: Optional[Sequence[Optional[torch.Tensor]]] = None,
        log_probs: Optional[Sequence[Optional[torch.Tensor]]] = None,
        pad_value: float = 0.0,
        context_pad_value: Optional[float] = None,
    ) -> "SequenceSet":
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of tensors.")

        B = len(sequences)
        if contexts is not None and len(contexts) != B:
            raise ValueError(
                f"contexts length {len(contexts)} does not match sequences length {B}"
            )
        if log_probs is not None and len(log_probs) != B:
            raise ValueError(
                f"log_probs length {len(log_probs)} does not match sequences length {B}"
            )

        seq_tensors = [s if s.ndim > 1 else s.unsqueeze(-1) for s in sequences]
        if any(t.ndim != 2 for t in seq_tensors):
            raise ValueError("each sequence must be 1D or 2D")
        feature_dim = seq_tensors[0].shape[1]
        if any(t.shape[1] != feature_dim for t in seq_tensors):
            raise ValueError("all sequences must have the same feature dimension")
        if any(t.device != seq_tensors[0].device for t in seq_tensors):
            raise ValueError("all sequences must be on the same device")
        if any(t.dtype != seq_tensors[0].dtype for t in seq_tensors):
            raise ValueError("all sequences must have the same dtype")

        lengths = torch.tensor(
            [s.shape[0] for s in sequences],
            dtype=torch.long,
            device=sequences[0].device,
        )
        T_max = max(s.shape[0] for s in sequences)
        F = feature_dim

        seq_tensor = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_value)
        mask_tensor = (
            torch.arange(T_max, device=seq_tensor.device).expand(B, T_max)
            < lengths.unsqueeze(1)
        ).unsqueeze(-1)

        ctx_dim = F
        if contexts is not None:
            context_tensors = [
                c if c is None or c.ndim > 1 else c.unsqueeze(-1)
                for c in contexts
            ]
            explicit_dims = {
                c.shape[1]
                for c in context_tensors
                if c is not None
            }
            if len(explicit_dims) > 1:
                raise ValueError("all explicit contexts must have the same feature dimension")
            if explicit_dims:
                ctx_dim = explicit_dims.pop()
            if any(c is None for c in context_tensors) and ctx_dim != F:
                raise ValueError(
                    "None context fallback requires context dimension to match sequence feature dimension"
                )
        else:
            context_tensors = None
        ctx_pad = context_pad_value if context_pad_value is not None else 0.0
        ctx_tensor = torch.full(
            (B, T_max, ctx_dim),
            ctx_pad,
            dtype=seq_tensor.dtype,
            device=seq_tensor.device,
        )

        if context_tensors is not None:
            for i, c in enumerate(context_tensors):
                L = int(lengths[i].item())
                if c is None:
                    ctx_tensor[i, :L] = seq_tensor[i, :L]
                    continue
                if c.ndim != 2:
                    raise ValueError("each context must be 1D or 2D")
                if c.shape[0] != L:
                    raise ValueError(
                        f"Context length {c.shape[0]} does not match sequence length {L}"
                    )
                if c.shape[1] != ctx_dim:
                    raise ValueError(
                        f"Context dimension {c.shape[1]} does not match expected {ctx_dim}"
                    )
                ctx_tensor[i, :L] = c.to(device=seq_tensor.device, dtype=seq_tensor.dtype)
        else:
            ctx_tensor[:, :, :F] = seq_tensor

        logp_tensor = None
        if log_probs is not None:
            logp_tensors = [
                lp if lp is None or lp.ndim > 1 else lp.unsqueeze(-1)
                for lp in log_probs
            ]
            state_dims = {
                lp.shape[1]
                for lp in logp_tensors
                if lp is not None
            }
            if len(state_dims) > 1:
                raise ValueError("all log_probs must have the same state dimension")
            K = state_dims.pop() if state_dims else 1
            logp_tensor = torch.full(
                (B, T_max, K),
                float("-inf"),
                dtype=seq_tensor.dtype,
                device=seq_tensor.device,
            )
            for i, lp in enumerate(logp_tensors):
                if lp is None:
                    continue
                if lp.ndim != 2:
                    raise ValueError("each log_probs tensor must be 1D or 2D")
                L = int(lengths[i].item())
                if lp.shape[0] != L:
                    raise ValueError(
                        f"log_probs length {lp.shape[0]} does not match sequence length {L}"
                    )
                logp_tensor[i, :L] = lp.to(
                    device=seq_tensor.device,
                    dtype=seq_tensor.dtype,
                )

        denom = mask_tensor.sum(dim=1).clamp_min(1)
        canonical_tensor = (ctx_tensor * mask_tensor).sum(dim=1) / denom
        canonical_tensor = canonical_tensor.unsqueeze(1)

        return cls(
            sequences=seq_tensor,
            lengths=lengths,
            masks=mask_tensor,
            contexts=ctx_tensor,
            canonical=canonical_tensor,
            log_probs=logp_tensor,
        )

    @property
    def mask(self) -> torch.BoolTensor:
        return self.masks

    @property
    def n_sequences(self) -> int:
        return self.sequences.shape[0]

    @property
    def total_timesteps(self) -> int:
        return int(self.lengths.sum())

    @property
    def feature_dim(self) -> int:
        return self.sequences.shape[2]

    @property
    def context_dim(self) -> int:
        return self.contexts.shape[2]

    def select(self, indices: torch.Tensor | list[int]) -> "SequenceSet":
        if isinstance(indices, list):
            indices = torch.tensor(
                indices, dtype=torch.long, device=self.sequences.device
            )
        return SequenceSet(
            sequences=self.sequences[indices],
            lengths=self.lengths[indices],
            masks=self.masks[indices],
            contexts=self.contexts[indices],
            canonical=self.canonical[indices],
            log_probs=self.log_probs[indices] if self.log_probs is not None else None,
        )

    @staticmethod
    def batchify(items: Sequence[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        if not items:
            raise ValueError("Cannot batchify empty list.")
        B = len(items)
        T_max = max(t.shape[0] for t in items)
        F = items[0].shape[1] if items[0].ndim > 1 else 1
        out = torch.full(
            (B, T_max, F),
            pad_value,
            dtype=items[0].dtype,
            device=items[0].device,
        )
        for i, t in enumerate(items):
            t_ = t if t.ndim > 1 else t.unsqueeze(-1)
            out[i, : t_.shape[0], : t_.shape[1]] = t_
        return out

    def update(
        self,
        encoder: "ContextEncoder",
        pool: Optional[str] = None,
        detach: bool = False,
    ) -> None:
        mask = self.masks.squeeze(-1)
        with torch.no_grad() if detach else contextlib.nullcontext():
            contexts, canonical = encoder.encode(
                self.sequences,
                mask=mask,
                pool=pool,
            )

        if contexts.ndim != 3 or contexts.shape[:2] != self.sequences.shape[:2]:
            raise ValueError(
                "encoder sequence context must match SequenceSet batch/time dimensions"
            )
        if canonical.ndim != 3 or canonical.shape[:2] != (self.n_sequences, 1):
            raise ValueError("encoder canonical context must be [B,1,H]")

        self.contexts = contexts
        self.canonical = canonical


@dataclass
class ContextRouter:
    context: torch.Tensor
    canonical: torch.Tensor
    names: Optional[list[str]] = None
    mask: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    _cache: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.canonical.ndim != 3 or self.canonical.shape[1] != 1:
            raise ValueError(
                f"canonical must be [B,1,H], got {tuple(self.canonical.shape)}"
            )
        if self.context.ndim != 3:
            raise ValueError(f"context must be [B,T,H], got {tuple(self.context.shape)}")

        B, T, H = self.context.shape
        if self.canonical.shape[0] != B or self.canonical.shape[2] != H:
            raise ValueError("canonical incompatible with context")

        if self.names is None:
            self.names = [f"context{i}" for i in range(H)]
        elif len(self.names) != H:
            raise ValueError(f"names length {len(self.names)} != H={H}")

        if self.mask is None:
            self.mask = torch.ones(
                B, T, 1, dtype=torch.bool, device=self.context.device
            )
        else:
            self.mask = self.mask.to(device=self.context.device, dtype=torch.bool)
            if self.mask.ndim == 2:
                self.mask = self.mask.unsqueeze(-1)
            if self.mask.shape[0] != B:
                raise ValueError(f"mask batch mismatch: {self.mask.shape[0]} != {B}")
            if self.mask.shape[1] != T:
                raise ValueError(f"mask time mismatch: {self.mask.shape[1]} != {T}")
            if self.mask.shape[2] != 1:
                raise ValueError(f"mask last dim must be 1, got {self.mask.shape[2]}")

        if self.log_probs is not None:
            if self.log_probs.ndim != 3:
                raise ValueError(
                    f"log_probs must be [B,T,K], got {tuple(self.log_probs.shape)}"
                )
            if self.log_probs.shape[:2] != (B, T):
                raise ValueError(
                    "log_probs batch/time dimensions must match context: "
                    f"{tuple(self.log_probs.shape[:2])} != {(B, T)}"
                )
        self._cache.update({"B": B, "T": T, "H": H})

    @classmethod
    def from_tensor(
        cls,
        X: "SequenceSet",
        context: Optional[Union[torch.Tensor, "ContextRouter"]] = None,
        mode: str = "additive",
    ) -> "ContextRouter":
        if not isinstance(X, SequenceSet):
            raise TypeError("X must be a SequenceSet")

        B, T, H = X.n_sequences, X.sequences.shape[1], X.context_dim
        ctx = X.contexts.clone()
        canonical = X.canonical.clone()
        mask = X.masks.clone()
        log_probs = X.log_probs.clone() if X.log_probs is not None else None
        names = [f"context{i}" for i in range(H)]

        ctx_override = context.context if isinstance(context, ContextRouter) else context
        if ctx_override is not None:
            ctx_override = align_context_tensor(
                ctx_override,
                batch_size=B,
                timesteps=T,
                context_dim=H,
                device=ctx.device,
                dtype=ctx.dtype,
            )

            if mode == "additive":
                ctx = ctx + ctx_override
                canonical = canonical + ctx_override[:, :1]
            elif mode == "replace":
                ctx = ctx_override
                canonical = ctx_override[:, :1]
            else:
                raise ValueError(f"Unsupported mode {mode}")

        return cls(
            canonical=canonical,
            context=ctx,
            mask=mask,
            log_probs=log_probs,
            names=names,
        )

    def get_context(self):
        return self.context

    def get_canonical(self):
        return self.canonical

    def get_feature_names(self):
        return self.names

    def get_log_probs(self):
        return self.log_probs

    def get_mask(self):
        return self.mask

    def select_features(self, keys: list[str]) -> "ContextRouter":
        idx = torch.tensor(
            [self.names.index(k) for k in keys], device=self.context.device
        )
        return ContextRouter(
            canonical=self.canonical[:, :, idx],
            context=self.context[:, :, idx],
            names=keys,
            mask=self.mask,
            log_probs=self.log_probs,
        )

    def select(self, indices: torch.Tensor | list[int]) -> "ContextRouter":
        if isinstance(indices, list):
            indices = torch.tensor(
                indices, dtype=torch.long, device=self.context.device
            )
        return ContextRouter(
            canonical=self.canonical[indices],
            context=self.context[indices],
            names=self.names.copy(),
            mask=self.mask[indices],
            log_probs=self.log_probs[indices] if self.log_probs is not None else None,
        )

    def detach(self) -> "ContextRouter":
        return ContextRouter(
            canonical=self.canonical.detach(),
            context=self.context.detach(),
            names=self.names,
            mask=self.mask.detach(),
            log_probs=self.log_probs.detach() if self.log_probs is not None else None,
        )

    def clone(self) -> "ContextRouter":
        return ContextRouter(
            canonical=self.canonical.clone(),
            context=self.context.clone(),
            names=self.names.copy(),
            mask=self.mask.clone(),
            log_probs=self.log_probs.clone() if self.log_probs is not None else None,
        )


class ContextEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_heads: int = 4,
        dropout: float = 0.0,
        layer_norm: bool = True,
        context_scale: float = 1.0,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
    ):
        super().__init__()
        self.pool = pool
        self.n_heads = n_heads
        self.encoder = encoder
        self.layer_norm = layer_norm
        self.context_scale = context_scale
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._context: Optional[torch.Tensor] = None
        self._sequence: Optional[torch.Tensor] = None
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None
        self._POOLERS: Dict[str, Callable] = {
            "mean": self._pool_mean,
            "last": self._pool_last,
            "max": self._pool_max,
            "attn": self._attention_context,
            "mha": self._multihead_context,
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        return_attn_weights: bool = False,
        return_context: bool = False,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, _ = x.shape
        mask = self._prepare_mask(mask, B, T, device=x.device)

        if "mask" in self._encoder_signature():
            out = self.encoder(x, mask=mask)
        else:
            out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        context = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        self._sequence = context
        pooled, attn = self._pool_context(context, mask, return_attn_weights)
        if self.layer_norm:
            pooled = nnF.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled * self.context_scale))
        canonical = pooled.unsqueeze(1)
        self._context = canonical
        seq_out = context if return_sequence else self._last_timestep(context, mask)
        return (
            seq_out,
            canonical if return_context else None,
            attn if return_attn_weights else None,
        )

    def encode(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_pool = self.pool
        if pool is not None:
            self.pool = pool
        try:
            seq_features, canonical, _ = self.forward(
                sequences, mask=mask, return_sequence=True, return_context=True
            )
        finally:
            if pool is not None:
                self.pool = old_pool
        return seq_features, canonical

    def _prepare_mask(
        self,
        mask: Optional[torch.BoolTensor],
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        if mask is None:
            return torch.ones(B, T, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)
        elif mask.ndim == 3:
            mask = mask.squeeze(-1)
        if mask.shape != (B, T):
            raise ValueError(f"Mask shape {mask.shape} does not match input shape {(B, T)}")
        return mask

    def _last_timestep(
        self, context: torch.Tensor, mask: Optional[torch.BoolTensor]
    ) -> torch.Tensor:
        if mask is not None:
            idx = torch.clamp(mask.sum(dim=1) - 1, min=0)
            return context[
                torch.arange(context.shape[0], device=context.device), idx
            ]
        return context[:, -1, :]

    def _pool_context(
        self, context: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool
    ):
        if self.pool not in self._POOLERS:
            raise ValueError(f"Invalid pooling method '{self.pool}'")
        return self._POOLERS[self.pool](context, mask, ret_attn)

    def _pool_mean(self, context, mask, ret_attn):
        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            ctx = context.sum(dim=1) / denom
        else:
            ctx = context.mean(dim=1)
        return ctx, None

    def _pool_last(self, context, mask, ret_attn):
        return self._last_timestep(context, mask), None

    def _pool_max(self, context, mask, ret_attn):
        if mask is not None:
            masked = context.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            ctx = masked.max(dim=1).values
            ctx = torch.where(torch.isfinite(ctx), ctx, torch.zeros_like(ctx))
        else:
            ctx = context.max(dim=1).values
        return ctx, None

    def _init_attn_vector(
        self, H: int, *, device: torch.device, dtype: torch.dtype
    ):
        if self._attn_vector is None:
            self._attn_vector = nn.Parameter(
                torch.randn(H, device=device, dtype=dtype) * 0.1
            )
            self.register_parameter("_attn_vector", self._attn_vector)
        elif self._attn_vector.shape != (H,):
            raise ValueError(
                f"attention pooling dimension changed from {self._attn_vector.shape[0]} to {H}"
            )

    def _attention_context(self, context, mask, ret_attn):
        _, _, H = context.shape
        self._init_attn_vector(H, device=context.device, dtype=context.dtype)
        scores = context @ self._attn_vector
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_w = nnF.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (attn_w * context).sum(dim=1)
        return pooled, attn_w if ret_attn else None

    def _multihead_context(self, context, mask, ret_attn):
        _, _, H = context.shape
        if H % self.n_heads != 0:
            raise ValueError(
                f"mha pooling requires context dimension {H} to be divisible by n_heads={self.n_heads}"
            )
        if self._mha is None:
            self._mha = nn.MultiheadAttention(
                embed_dim=H,
                num_heads=self.n_heads,
                batch_first=True,
                dropout=(
                    self.dropout_layer.p
                    if isinstance(self.dropout_layer, nn.Dropout)
                    else 0.0
                ),
            ).to(device=context.device, dtype=context.dtype)
        elif self._mha.embed_dim != H:
            raise ValueError(
                f"mha pooling dimension changed from {self._mha.embed_dim} to {H}"
            )
        key_padding_mask = ~mask if mask is not None else None
        out, attn = self._mha(
            context, context, context, key_padding_mask=key_padding_mask
        )
        pooled = out.mean(dim=1)
        return pooled, attn if ret_attn else None

    def reset(self):
        self._sequence = None
        self._context = None

    def _encoder_signature(self) -> Tuple[str, ...]:
        try:
            return tuple(inspect.signature(self.encoder.forward).parameters)
        except (TypeError, ValueError):
            return tuple()
