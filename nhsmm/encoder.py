# nhsmm/encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnF


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
        self.padding = 0
        self.bidirectional = False if self.causal else bidirectional
        self.return_sequence = return_sequence
        self.use_packed = use_packed

        self.conv = nn.Conv1d(n_features, cnn_channels, cnn_kernel, padding=self.padding)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.norm = nn.LayerNorm(cnn_channels)
        self.lstm = nn.LSTM(
            cnn_channels, hidden_dim, batch_first=True, bidirectional=self.bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if self.bidirectional else 1)

        self._context = None
        self.register_buffer("_stream_lstm_weight", None, persistent=False)
        self.register_buffer("_stream_lstm_bias", None, persistent=False)
        self._stream_lstm_cache_key = None

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

        # In streaming mode ``conv_input`` is exactly one causal convolution
        # window, so Conv1d would dispatch a general convolution only to emit
        # one output. The equivalent flattened linear form reuses the same
        # Conv1d weights/bias without changing parameters or artifact state.
        conv_flat = conv_input.transpose(1, 2).reshape(B, -1)
        conv_weight = self.conv.weight.reshape(self.cnn_channels, -1)
        x_c = nnF.linear(conv_flat, conv_weight, self.conv.bias).unsqueeze(1)
        x_c = nnF.relu(x_c)
        x_c = self.norm(x_c)

        # ``stream_step`` always processes exactly one timestep. Calling
        # ``nn.LSTM`` for a length-one sequence pays the generic recurrent
        # sequence dispatcher cost on every bar, so evaluate the same single
        # layer LSTM equations directly with the existing LSTM parameters.
        x_t = x_c[:, 0]
        hidden_prev = state.hidden[0]
        cell_prev = state.cell[0]
        if torch.is_grad_enabled():
            gates = nnF.linear(
                x_t,
                self.lstm.weight_ih_l0,
                self.lstm.bias_ih_l0,
            ) + nnF.linear(
                hidden_prev,
                self.lstm.weight_hh_l0,
                self.lstm.bias_hh_l0,
            )
        else:
            parameters = (
                self.lstm.weight_ih_l0,
                self.lstm.weight_hh_l0,
                self.lstm.bias_ih_l0,
                self.lstm.bias_hh_l0,
            )
            cache_key = tuple(int(parameter._version) for parameter in parameters)
            if (
                self._stream_lstm_weight is None
                or self._stream_lstm_bias is None
                or cache_key != self._stream_lstm_cache_key
            ):
                self._stream_lstm_weight = torch.cat(
                    (self.lstm.weight_ih_l0, self.lstm.weight_hh_l0),
                    dim=1,
                )
                self._stream_lstm_bias = self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0
                self._stream_lstm_cache_key = cache_key
            recurrent_input = torch.cat((x_t, hidden_prev), dim=-1)
            gates = nnF.linear(
                recurrent_input,
                self._stream_lstm_weight,
                self._stream_lstm_bias,
            )
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
        cell_t = torch.sigmoid(forget_gate) * cell_prev + torch.sigmoid(input_gate) * torch.tanh(
            cell_gate
        )
        hidden_t = torch.sigmoid(output_gate) * torch.tanh(cell_t)
        hidden = hidden_t.unsqueeze(0)
        cell = cell_t.unsqueeze(0)
        out = hidden_t.unsqueeze(1)

        if self.cnn_kernel > 1:
            history = conv_input[:, -(self.cnn_kernel - 1) :]
        else:
            history = x.new_empty(B, 0, self.n_features)

        return out, DefaultEncoderStreamState(
            conv_history=history,
            hidden=hidden,
            cell=cell,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_sequence: bool | None = None,
    ):
        # canonicalize input
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError("encoder input must be [F], [T,F], or [B,T,F]")
        B, T, F_in = x.shape
        if F_in != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {F_in}")
        if T == 0:
            raise ValueError("Input sequence has zero length")

        # canonicalize mask
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.ndim == 1:
                if mask.shape[0] != T:
                    raise ValueError(
                        f"mask length {mask.shape[0]} does not match sequence length {T}"
                    )
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.ndim == 2:
                if mask.shape[0] == 1 and B > 1:
                    mask = mask.expand(B, -1)
                if mask.shape != (B, T):
                    raise ValueError(f"mask shape {tuple(mask.shape)} does not match {(B, T)}")
            else:
                raise ValueError("mask must have shape [T] or [B,T]")
            if T > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
                raise ValueError(
                    "mask must be right-padded (valid timesteps followed only by padding)"
                )

        # Masked timesteps are padding, not observations. Zero them before
        # convolution so non-causal kernels cannot leak padded values backward
        # into valid timesteps near the sequence boundary.
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        # --- CNN ---
        x_c = x.transpose(1, 2)  # [B,F,T]
        if self.cnn_kernel > 1:
            if self.causal:
                x_c = nnF.pad(x_c, (self.cnn_kernel - 1, 0))
            else:
                total_pad = self.cnn_kernel - 1
                left = total_pad // 2
                right = total_pad - left
                x_c = nnF.pad(x_c, (left, right))
        x_c = nnF.relu(self.conv(x_c))
        x_c = x_c.transpose(1, 2)  # [B,T,F]
        x_c = self.norm(x_c)
        x_c = self.dropout(x_c)

        # --- LSTM ---
        if self.use_packed and mask is not None:
            lengths = mask.sum(dim=1)
            nonempty = lengths > 0
            out = x_c.new_zeros(B, T, self.out_dim)
            if bool(nonempty.any()):
                packed = nn.utils.rnn.pack_padded_sequence(
                    x_c[nonempty],
                    lengths[nonempty].cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                out_packed, _ = self.lstm(packed)
                unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                    out_packed,
                    batch_first=True,
                    total_length=T,
                )
                out[nonempty] = unpacked
        else:
            out, _ = self.lstm(x_c)
        out = self.dropout(out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        # --- Pooled canonical context ---
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(dtype=out.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (out * mask_f).sum(dim=1) / denom
        else:
            pooled = out.mean(dim=1)
        self._context = pooled.detach().unsqueeze(1)  # [B,1,H], diagnostic cache only

        # --- Return sequence or last valid timestep ---
        ret_seq = self.return_sequence if return_sequence is None else bool(return_sequence)
        if ret_seq:
            return out
        if mask is not None:
            lengths = mask.sum(dim=1)
            idx = lengths.clamp_min(1) - 1
            last = out[torch.arange(B, device=out.device), idx]
            return torch.where(lengths.unsqueeze(-1) > 0, last, torch.zeros_like(last))
        return out[:, -1, :]
