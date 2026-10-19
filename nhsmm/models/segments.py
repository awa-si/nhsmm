"""Support-safe segment emission sums. doc:nhsmm.segment-emissions"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EmissionPrefix:
    values: torch.Tensor
    impossible: torch.Tensor | None

    @classmethod
    def build(cls, log_prob: torch.Tensor, dim: int) -> "EmissionPrefix":
        bad = torch.isneginf(log_prob)
        if torch.isnan(log_prob).any() or torch.isposinf(log_prob).any():
            raise ValueError("emission scores must not contain NaN or +inf")
        shape = list(log_prob.shape)
        shape[dim] = 1
        zero = log_prob.new_zeros(shape)
        has_impossible = bool(bad.any())
        finite = log_prob.masked_fill(bad, 0.0) if has_impossible else log_prob
        values = torch.cat((zero, finite.cumsum(dim)), dim=dim)
        impossible = None
        if has_impossible:
            counts = bad.to(torch.long)
            impossible = torch.cat((counts.new_zeros(shape), counts.cumsum(dim)), dim=dim)
        return cls(values, impossible)

    def segment(self, end, start) -> torch.Tensor:
        result = self.values[end] - self.values[start]
        if self.impossible is not None:
            count = self.impossible[end] - self.impossible[start]
            result = result.masked_fill(count > 0, float("-inf"))
        return result

    def gathered(self, end: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Batched [B,T,K,D] endpoint indices for a time-axis-one prefix."""
        values = self.values.unsqueeze(-1).expand(-1, -1, -1, end.shape[-1])
        result = values.gather(1, end) - values.gather(1, start)
        if self.impossible is not None:
            counts = self.impossible.unsqueeze(-1).expand_as(values)
            count = counts.gather(1, end) - counts.gather(1, start)
            result = result.masked_fill(count > 0, float("-inf"))
        return result
