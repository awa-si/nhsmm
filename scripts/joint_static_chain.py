"""Research-only all-static Gaussian/duration/transition/initial censored EM.

Separate owner; prior frozen known-transition studies remain byte-identical.
Structural zero diagonal, positive full duration support; no latent labels.
"""

import numpy as np

from scripts.block_mass_smoothing import marginal_objective
from scripts.joint_gaussian_duration import joint_mstep, scores

MASK = ~np.eye(4, dtype=bool)


def validate_laws(transition, initial):
    transition, initial = np.asarray(transition, dtype=float), np.asarray(initial, dtype=float)
    if (
        transition.shape != (4, 4)
        or initial.shape != (4,)
        or not np.isfinite(transition).all()
        or not np.isfinite(initial).all()
        or (transition[MASK] <= 0).any()
        or (transition[~MASK] != 0).any()
        or (initial <= 0).any()
        or not np.allclose(transition.sum(1), 1, atol=1e-12, rtol=0)
        or not np.isclose(initial.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("positive normalized static laws with structural zero diagonal required")
    return transition, initial


def chunk_statistics(x, pmf, means, variances, transition, initial, *, return_filter=False):
    """Forward/backward with censored endpoint; soft occupancy and episode counts."""
    transition, initial = validate_laws(transition, initial)
    emission = scores(x, means, variances)
    pmf = np.asarray(pmf, dtype=float)
    batch, steps, states = emission.shape
    if (
        pmf.ndim != 2
        or pmf.shape[0] != states
        or not pmf.shape[1]
        or not np.isfinite(pmf).all()
        or (pmf <= 0).any()
        or not np.allclose(pmf.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("positive normalized duration PMF required")
    width = pmf.shape[1]
    survival = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    hazard = pmf / survival
    continuation = np.pad(survival[:, 1:], ((0, 0), (0, 1))) / survival
    offset = emission.max(2)
    relative = np.exp(emission - offset[:, :, None])
    alpha = np.zeros((batch, steps, states, width))
    log_z = np.empty((batch, steps))
    scales = np.empty((batch, steps))
    frontier = np.zeros((batch, states, width))
    frontier[:, :, 0] = initial
    for t in range(steps):
        if t:
            predicted = np.zeros_like(frontier)
            predicted[:, :, 0] = (frontier * hazard).sum(2) @ transition
            predicted[:, :, 1:] = frontier[:, :, :-1] * continuation[:, :-1]
            frontier = predicted
        frontier = frontier * relative[:, t, :, None]
        normalizer = frontier.sum((1, 2))
        if (normalizer <= 0).any() or not np.isfinite(normalizer).all():
            raise FloatingPointError("impossible/nonfinite Gaussian evidence")
        log_z[:, t] = np.log(normalizer) + offset[:, t]
        scales[:, t] = normalizer
        frontier /= normalizer[:, None, None]
        alpha[:, t] = frontier
    complete = np.zeros_like(pmf)
    censored = alpha[:, -1].sum(0)
    beta = np.ones((batch, states, width))
    occupancy = np.zeros(states)
    first = np.zeros(states)
    second = np.zeros(states)
    transition_counts = np.zeros((states, states))
    initial_counts = np.zeros(states)

    def accumulate(t, gamma):
        if not np.isfinite(gamma).all() or not np.allclose(gamma.sum(1), 1, atol=1e-8, rtol=0):
            raise FloatingPointError("posterior state occupancy normalization")
        if t == 0:
            initial_counts[:] += gamma.sum(0)
        values = np.asarray(x[:, t, 0], dtype=float)
        occupancy[:] += gamma.sum(0)
        first[:] += (gamma * values[:, None]).sum(0)
        second[:] += (gamma * values[:, None] ** 2).sum(0)

    accumulate(steps - 1, alpha[:, -1].sum(2))
    for t in range(steps - 2, -1, -1):
        future = relative[:, t + 1] * beta[:, :, 0]
        exit_mass = (alpha[:, t] * hazard).sum(2) / scales[:, t + 1, None]
        transition_counts += (exit_mass[:, :, None] * transition[None] * future[:, None, :]).sum(0)
        boundary = future @ transition.T
        ending = hazard[None] * boundary[:, :, None]
        complete += (alpha[:, t] * ending / scales[:, t + 1, None, None]).sum(0)
        continuing = np.zeros_like(beta)
        continuing[:, :, :-1] = (
            continuation[None, :, :-1] * relative[:, t + 1, :, None] * beta[:, :, 1:]
        )
        beta = (ending + continuing) / scales[:, t + 1, None, None]
        if not np.isfinite(beta).all():
            raise FloatingPointError("nonfinite scaled Gaussian backward probabilities")
        accumulate(t, (alpha[:, t] * beta).sum(2))
    expected = complete + pmf * np.cumsum(censored / survival, axis=1)
    if not np.isfinite(expected).all():
        raise FloatingPointError("nonfinite duration statistics")
    np.testing.assert_allclose(expected.sum(), complete.sum() + batch, rtol=0, atol=1e-8)
    np.testing.assert_allclose(occupancy.sum(), batch * steps, rtol=0, atol=1e-8)
    np.testing.assert_allclose(transition_counts.sum(), complete.sum(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(initial_counts.sum(), batch, rtol=0, atol=1e-8)
    result = {
        "transition_counts": transition_counts,
        "initial_counts": initial_counts,
        "ll": float(log_z.sum()),
        "duration_counts": expected,
        "occupancy": occupancy,
        "sum_x": first,
        "sum_x2": second,
    }
    if return_filter:
        result["filtered_state_probabilities"] = alpha.sum(3)
    return result


def statistics(x, pmf, means, variances, transition, initial, *, chunk=16):
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("positive integer chunk required")
    validate_laws(transition, initial)
    scores(x, means, variances)  # Validate full input before any accumulation.
    result = None
    for start in range(0, len(x), chunk):
        part = chunk_statistics(
            x[start : start + chunk], pmf, means, variances, transition, initial
        )
        if result is None:
            result = part
        else:
            for key in result:
                result[key] += part[key]
    return result


def validate_prior(prior):
    if isinstance(prior, bool) or not np.isfinite(prior) or prior <= 0:
        raise ValueError("positive finite chain pseudocount required")


def objective(ll, pmf, transition, initial, *, chain_pseudocount=1e-6):
    transition, initial = validate_laws(transition, initial)
    validate_prior(chain_pseudocount)
    return marginal_objective(ll, pmf, 0.0) + chain_pseudocount * (
        np.log(transition[MASK]).sum() + np.log(initial).sum()
    )


def chain_gradients(stats, transition, initial, *, chain_pseudocount=1e-6):
    transition, initial = validate_laws(transition, initial)
    validate_prior(chain_pseudocount)
    counts = np.asarray(stats["transition_counts"]) + chain_pseudocount * MASK
    first = np.asarray(stats["initial_counts"]) + chain_pseudocount
    return counts - counts.sum(1, keepdims=True) * transition, first - first.sum() * initial


def full_mstep(stats, *, chain_pseudocount=1e-6, width=8, variance_floor=1e-6):
    validate_prior(chain_pseudocount)
    counts = np.asarray(stats["transition_counts"], dtype=float)
    first = np.asarray(stats["initial_counts"], dtype=float)
    if (
        counts.shape != (4, 4)
        or first.shape != (4,)
        or not np.isfinite(counts).all()
        or not np.isfinite(first).all()
        or (counts < 0).any()
        or (first < 0).any()
        or (counts[~MASK] != 0).any()
    ):
        raise ValueError("nonnegative finite structural chain counts required")
    augmented = counts + chain_pseudocount * MASK
    transition = augmented / augmented.sum(1, keepdims=True)
    initial = (first + chain_pseudocount) / (first.sum() + 4 * chain_pseudocount)
    validate_laws(transition, initial)
    p, means, variances, certificate = joint_mstep(
        stats, width=width, variance_floor=variance_floor
    )
    return p, means, variances, transition, initial, certificate
