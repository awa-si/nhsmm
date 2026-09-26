# nhsmm/encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from nhsmm.config import logger


@dataclass(frozen=True)
class DefaultEncoderStreamState:
    """Bounded causal state for one-timestep DefaultEncoder updates."""

    conv_history: torch.Tensor  # [B, cnn_kernel-1, F]
    hidden: torch.Tensor  # [1,B,H]
    cell: torch.Tensor  # [1,B,H]


class DefaultEncoder(nn.Module):
    """
    Default CNN + LSTM encoder for sequences.
    Returns per-timestep features and pooled canonical context.

    When ``causal=True``, the CNN uses left-only padding and the LSTM is
    unidirectional so timestep ``t`` cannot consume observations after ``t``.
    """

    def __init__(
        self,
        n_features: int,
        cnn_kernel: int = 3,
        hidden_dim: int = 32,
        cnn_channels: int = 16,
        bidirectional: bool = True,
        return_sequence: bool = True,
        use_packed: bool = True,
        dropout: float = 0.05,
        causal: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.cnn_channels = cnn_channels
        self.cnn_kernel = cnn_kernel
        self.causal = bool(causal)
        self.padding = 0 if self.causal else cnn_kernel // 2
        self.bidirectional = False if self.causal else bidirectional
        self.return_sequence = return_sequence
        self.use_packed = use_packed

        self.conv = nn.Conv1d(n_features, cnn_channels, cnn_kernel, padding=self.padding)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.norm = nn.LayerNorm(cnn_channels)
        self.lstm = nn.LSTM(
            cnn_channels,
            hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if self.bidirectional else 1)

        self._context = None

    def initial_stream_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DefaultEncoderStreamState:
        """Create zero causal CNN/LSTM state for a streaming batch."""

        if not self.causal:
            raise RuntimeError("incremental encoder state requires causal=True")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        reference = self.conv.weight
        device = reference.device if device is None else torch.device(device)
        dtype = reference.dtype if dtype is None else dtype
        history = torch.zeros(
            batch_size,
            max(self.cnn_kernel - 1, 0),
            self.n_features,
            device=device,
            dtype=dtype,
        )
        hidden = torch.zeros(
            1,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        cell = torch.zeros_like(hidden)
        return DefaultEncoderStreamState(
            conv_history=history,
            hidden=hidden,
            cell=cell,
        )

    def stream_step(
        self,
        x: torch.Tensor,
        state: Optional[DefaultEncoderStreamState] = None,
    ) -> tuple[torch.Tensor, DefaultEncoderStreamState]:
        """Encode exactly one causal timestep without re-encoding its prefix."""

        if not self.causal:
            raise RuntimeError("incremental encoder state requires causal=True")
        if self.training:
            raise RuntimeError("incremental encoder state requires eval mode")

        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("stream_step input must be [F], [B,F], or [B,1,F]")
        B, T, F_in = x.shape
        if T != 1:
            raise ValueError("stream_step accepts exactly one timestep")
        if F_in != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {F_in}")

        if state is None:
            state = self.initial_stream_state(
                B,
                device=x.device,
                dtype=x.dtype,
            )
        if not isinstance(state, DefaultEncoderStreamState):
            raise TypeError("state must be DefaultEncoderStreamState")

        expected_history = (B, max(self.cnn_kernel - 1, 0), self.n_features)
        expected_recurrent = (1, B, self.hidden_dim)
        if state.conv_history.shape != expected_history:
            raise ValueError(
                f"conv_history must be {expected_history}, got {state.conv_history.shape}"
            )
        if state.hidden.shape != expected_recurrent or state.cell.shape != expected_recurrent:
            raise ValueError(
                f"hidden/cell must be {expected_recurrent}, got "
                f"{state.hidden.shape}/{state.cell.shape}"
            )
        for name, tensor in (
            ("conv_history", state.conv_history),
            ("hidden", state.hidden),
            ("cell", state.cell),
        ):
            if tensor.device != x.device:
                raise ValueError(f"{name} device {tensor.device} != input device {x.device}")
            if tensor.dtype != x.dtype:
                raise ValueError(f"{name} dtype {tensor.dtype} != input dtype {x.dtype}")

        conv_input = torch.cat((state.conv_history, x), dim=1)
        x_c = conv_input.transpose(1, 2)
        x_c = nnF.relu(self.conv(x_c))
        x_c = x_c.transpose(1, 2)
        x_c = self.norm(x_c)
        x_c = self.dropout(x_c)

        out, (hidden, cell) = self.lstm(x_c, (state.hidden, state.cell))
        out = self.dropout(out)

        if self.cnn_kernel > 1:
            history = conv_input[:, -(self.cnn_kernel - 1):].clone()
        else:
            history = x.new_empty(B, 0, self.n_features)

        return out, DefaultEncoderStreamState(
            conv_history=history,
            hidden=hidden,
            cell=cell,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_sequence: bool | None = None):
        # canonicalize input
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, F_in = x.shape
        if F_in != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {F_in}")
        if T == 0:
            raise ValueError("Input sequence has zero length")

        # canonicalize mask
        if mask is not None:
            mask = mask.bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.shape[0] == 1 and B > 1:
                mask = mask.expand(B, -1)
            mask = mask[:, :T]

        # --- CNN ---
        x_c = x.transpose(1, 2)  # [B,F,T]
        if self.causal and self.cnn_kernel > 1:
            x_c = nnF.pad(x_c, (self.cnn_kernel - 1, 0))
        x_c = nnF.relu(self.conv(x_c))
        x_c = x_c.transpose(1, 2)  # [B,T,F]
        x_c = self.norm(x_c)
        x_c = self.dropout(x_c)

        # --- LSTM ---
        if self.use_packed and mask is not None:
            lengths = mask.sum(dim=1).clamp_min(1)
            packed = nn.utils.rnn.pack_padded_sequence(x_c, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x_c)
        out = self.dropout(out)

        # --- Pooled canonical context ---
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(dtype=out.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (out * mask_f).sum(dim=1) / denom
        else:
            pooled = out.mean(dim=1)
        self._context = pooled.unsqueeze(1)  # [B,1,H]

        # --- Return sequence or last valid timestep ---
        ret_seq = self.return_sequence if return_sequence is None else bool(return_sequence)
        if ret_seq:
            return out
        if mask is not None:
            idx = mask.sum(dim=1).clamp_min(1) - 1
            return out[torch.arange(B, device=out.device), idx]
        return out[:, -1, :]
