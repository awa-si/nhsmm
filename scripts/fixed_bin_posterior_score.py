"""Exact research score for fixed-bin duration-only, frozen-nuisance HSMMs.

Expected episode counts include censored terminal survival contributions from
the independent E-step. This is not a joint/neural optimizer shortcut.
"""

import numpy as np
import torch

from scripts.block_mass_smoothing import geometry


def density(logits, size=260, width=8):
    """Anchored last-bin log density zero; positive PMF over every age."""
    _, lengths, _ = geometry(size, width)
    if logits.ndim != 2 or logits.shape[1] != len(lengths) - 1 or not torch.isfinite(logits).all():
        raise ValueError("finite anchored bin logits required")
    z = torch.cat((logits, logits.new_zeros((len(logits), 1))), dim=1)
    widths = torch.tensor(lengths, device=logits.device, dtype=logits.dtype)
    logp = z - torch.logsumexp(z + widths.log(), dim=1, keepdim=True)
    p = torch.repeat_interleave(logp.exp(), torch.tensor(lengths, device=logits.device), dim=1)
    if not torch.isfinite(p).all() or not (p > 0).all():
        raise FloatingPointError("nonpositive or nonfinite duration density")
    return p


def score_gradient(logits, counts, *, size=260, width=8, pseudocount=1e-6, strength=0.0):
    """Marginal LL + age pseudocount prior - curvature penalty gradient.

    The Fisher identity maps expected episode counts to the observed-data score.
    No differentiation through the E-step, labels, true PMFs or oracle support.
    """
    counts = np.asarray(counts, dtype=float)
    if (
        counts.shape != (len(logits), size)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.isfinite(pseudocount)
        or pseudocount < 0
        or not np.isfinite(strength)
        or strength < 0
    ):
        raise ValueError("finite nonnegative matching counts/penalties required")
    blocks, lengths, curvature = geometry(size, width)
    with torch.no_grad():
        p = density(logits, size, width)
        augmented = torch.as_tensor(
            np.add.reduceat(counts + pseudocount, [a for a, _ in blocks], axis=1),
            device=logits.device,
            dtype=logits.dtype,
        )
        mass = p[:, [a for a, _ in blocks]] * torch.tensor(
            lengths, device=logits.device, dtype=logits.dtype
        )
        result = augmented - augmented.sum(1, keepdim=True) * mass
        z = torch.cat((logits, logits.new_zeros((len(logits), 1))), dim=1)
        r = torch.tensor(curvature, device=logits.device, dtype=logits.dtype)
        result = result - strength * (z @ r.T @ r)
        return result[:, :-1]
