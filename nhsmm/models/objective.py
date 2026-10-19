"""Normalized observation scores versus raw forward segment potentials.

doc:nhsmm.likelihood (docs/hsmm-mathematical-contract.md)
"""

from dataclasses import replace

import torch

from nhsmm.filtering import _safe_logsumexp
from nhsmm.models.boundary import tail_boundary_score


def raw_sequence_scores(model, sequence, temperature=None):
    alpha = (
        model.forward(sequence)
        if temperature is None
        else model.forward(sequence, temperature=temperature)
    )
    scores = alpha.new_zeros(len(sequence.sequences))
    valid = sequence.lengths > 0
    if bool(valid.any()):
        last = alpha[valid, sequence.lengths[valid] - 1]
        scores[valid] = _safe_logsumexp(last.flatten(1), dim=1)
    return tail_boundary_score(model, sequence, scores, temperature)


def observation_log_likelihood(model, sequence, temperature=None):
    numerator = raw_sequence_scores(model, sequence, temperature)
    if model.config.causal:
        return numerator
    # Fixed covariates and normalized per-row emission densities imply that
    # integrating out observations replaces every emission factor by one.
    unit_emissions = replace(sequence, log_probs=torch.zeros_like(sequence.log_probs))
    partition = raw_sequence_scores(model, unit_emissions, temperature)
    if not torch.isfinite(partition).all():
        raise ValueError("no completed segmentation exists for the supplied length and context")
    return numerator - partition
