"""Tiny explicit parameter-to-path oracle, independent of nhsmm imports/kernels.

Scope: declared tanh/projection/LayerNorm Gaussian fixtures, ergodic transitions,
causal HMM or no-self-reset HSMM, finite duration or D+ tail. Raw tensors only.
"""

import math
from itertools import product

import torch


def _correction(parameters, context, shape, scale_limit=0.5, emission=False):
    """Evaluate the declared small MLP algebraically, never call a model/module."""
    if context is None:
        return parameters["delta_scale"].new_zeros(shape)
    value = context
    if "_proj.weight" in parameters:
        value = value @ parameters["_proj.weight"].T + parameters["_proj.bias"]
    value = value @ parameters["context_net.0.weight"].T + parameters["context_net.0.bias"]
    centered = value - value.mean()
    value = centered / torch.sqrt((centered**2).mean() + 1e-5)
    value = value * parameters["context_net.1.weight"] + parameters["context_net.1.bias"]
    value = torch.tanh(value)
    value = value @ parameters["context_net.3.weight"].T + parameters["context_net.3.bias"]
    delta = torch.clamp(
        torch.tanh(value).reshape(shape) * parameters["delta_scale"], -scale_limit, scale_limit
    )
    if emission:
        delta = delta - delta.mean(dim=-2, keepdim=True)
        magnitude = delta.abs().amax(dim=-2, keepdim=True)
        delta = delta * torch.minimum(
            torch.ones_like(magnitude),
            scale_limit / magnitude.clamp_min(torch.finfo(delta.dtype).eps),
        )
    return delta


def _categorical(raw, correction, log_temperature):
    # Fixture categorical scores stay far from the package's +/-30 score cap.
    return (raw + correction) / log_temperature.exp()


def ingredients(
    parameters,
    x,
    emission_context,
    boundary_context,
    available,
    duration_gate,
    duration,
    tail,
    allow_self,
):
    """Direct links from raw weights, independent E/U routing and density formula."""
    k, features = parameters["emission"]["mu"].shape
    initial = parameters["initial"]
    e0 = emission_context[0] if available[0] else None
    pi_scores = _categorical(
        initial["logits"], _correction(initial, e0, (k,)), initial["log_temperature"]
    )
    pi = pi_scores.softmax(-1)
    emissions, durations, transitions = [], [], []
    for t, row in enumerate(x):
        e = emission_context[t] if available[t] else None
        u = boundary_context[t] if available[t] else None
        p = parameters["emission"]
        location = p["mu"] + _correction(p, e, (k, features), emission=True)
        variance = torch.log1p(torch.exp(p["log_var"])).clamp_min(1e-6)
        density = torch.exp(
            -0.5
            * (((row - location) ** 2 / variance) + variance.log() + math.log(2 * math.pi)).sum(-1)
        )
        emissions.append(density)
        p = parameters["duration"]
        logits = _categorical(p["logits"], _correction(p, u, (k, duration)), p["log_temperature"])
        gate = (1 / (1 + (-duration_gate).exp())).log()
        durations.append((logits + gate).softmax(-1))
        p = parameters["transition"]
        logits = _categorical(
            p["logits"], _correction(p, u, (k, duration, k)), p["log_temperature"]
        )
        if not allow_self:
            logits = logits.masked_fill(torch.eye(k, dtype=torch.bool)[:, None, :], -torch.inf)
        transitions.append(logits.softmax(-1))
    hazard = parameters["duration"]["tail_end_probability"] if tail else None
    return pi, torch.stack(emissions), torch.stack(durations), torch.stack(transitions), hazard


def path_weight(path, pi, emission, q, transition, tail, *, hmm):
    """Direct probability product; no terminal exit factor (right censoring)."""
    d = q.shape[-1]
    weight = pi[path[0]] * emission[0, path[0]]
    age = 1
    for t in range(1, len(path)):
        previous, current = path[t - 1], path[t]
        if age > d and tail is None:
            return weight * 0, age
        index = min(age, d) - 1
        hazard = (
            tail[previous]
            if tail is not None and age >= d
            else q[t - 1, previous, index] / q[t - 1, previous, index:].sum()
        )
        if hmm or previous != current:
            weight = weight * hazard * transition[t - 1, previous, index, current]
            age = 1
        else:
            weight = weight * (1 - hazard)
            age += 1
        weight = weight * emission[t, current]
    if age > d and tail is None:
        return weight * 0, age
    return weight, age


def enumerate_weights(parts, *, hmm, length):
    pi, emission, q, transition, tail = parts
    return [
        (path, *path_weight(path, pi, emission, q, transition, tail, hmm=hmm))
        for path in product(range(len(pi)), repeat=length)
    ]
