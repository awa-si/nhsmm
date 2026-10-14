"""Research-only censored Gaussian/duration sufficient statistics and joint EM.

Static known transition/initial laws, scalar observations, positive full-age
fixed-bin durations. No labels, true-law support or neural context in fitting.
"""

import math

import numpy as np

from scripts.block_mass_smoothing import smooth_mstep
from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.verify_isolated_trading_hsmm import lse


def scores(x, means, variances):
    x, means, variances = (
        np.asarray(x, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(variances, dtype=float),
    )
    if (
        x.ndim != 3
        or x.shape[2] != 1
        or not x.shape[0]
        or not x.shape[1]
        or means.shape != (4,)
        or variances.shape != (4,)
        or (variances <= 0).any()
        or not all(np.isfinite(v).all() for v in (x, means, variances))
    ):
        raise ValueError("finite scalar observations and four positive Gaussian variances required")
    return -0.5 * (
        math.log(2 * math.pi) + np.log(variances) + (x[:, :, 0, None] - means) ** 2 / variances
    )


def chunk_statistics(x, pmf, means, variances, *, return_filter=False):
    """Forward/backward with censored endpoint; soft occupancy and episode counts."""
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
    with np.errstate(divide="ignore"):
        log_end, log_continue = np.log(hazard), np.log(continuation)
        log_transition = np.log(BASE_TRANSITION)
    offset = emission.max(2)
    relative = np.exp(emission - offset[:, :, None])
    alpha = np.zeros((batch, steps, states, width))
    log_z = np.empty((batch, steps))
    frontier = np.zeros((batch, states, width))
    frontier[:, :, 0] = 0.25
    for t in range(steps):
        if t:
            predicted = np.zeros_like(frontier)
            predicted[:, :, 0] = (frontier * hazard).sum(2) @ BASE_TRANSITION
            predicted[:, :, 1:] = frontier[:, :, :-1] * continuation[:, :-1]
            frontier = predicted
        frontier = frontier * relative[:, t, :, None]
        normalizer = frontier.sum((1, 2))
        if (normalizer <= 0).any() or not np.isfinite(normalizer).all():
            raise FloatingPointError("impossible/nonfinite Gaussian evidence")
        log_z[:, t] = np.log(normalizer) + offset[:, t]
        frontier /= normalizer[:, None, None]
        alpha[:, t] = frontier
    complete = np.zeros_like(pmf)
    censored = alpha[:, -1].sum(0)
    beta = np.zeros((batch, states, width))
    occupancy = np.zeros(states)
    first = np.zeros(states)
    second = np.zeros(states)

    def accumulate(t, gamma):
        if not np.isfinite(gamma).all() or not np.allclose(gamma.sum(1), 1, atol=1e-8, rtol=0):
            raise FloatingPointError("posterior state occupancy normalization")
        values = np.asarray(x[:, t, 0], dtype=float)
        occupancy[:] += gamma.sum(0)
        first[:] += (gamma * values[:, None]).sum(0)
        second[:] += (gamma * values[:, None] ** 2).sum(0)

    accumulate(steps - 1, alpha[:, -1].sum(2))
    for t in range(steps - 2, -1, -1):
        future = emission[:, t + 1] + beta[:, :, 0]
        boundary = lse(log_transition[None] + future[:, None, :], 2)
        with np.errstate(divide="ignore"):
            log_alpha = np.log(alpha[:, t])
        events = log_alpha + log_end[None] + boundary[:, :, None] - log_z[:, t + 1, None, None]
        complete += np.exp(events).sum(0)
        continuing = np.full_like(beta, -np.inf)
        continuing[:, :, :-1] = (
            log_continue[None, :, :-1] + emission[:, t + 1, :, None] + beta[:, :, 1:]
        )
        beta = np.logaddexp(log_end[None] + boundary[:, :, None], continuing)
        beta -= log_z[:, t + 1, None, None]
        accumulate(t, np.exp(lse(log_alpha + beta, 2)))
    expected = complete + pmf * np.cumsum(censored / survival, axis=1)
    if not np.isfinite(expected).all():
        raise FloatingPointError("nonfinite duration statistics")
    np.testing.assert_allclose(expected.sum(), complete.sum() + batch, rtol=0, atol=1e-8)
    np.testing.assert_allclose(occupancy.sum(), batch * steps, rtol=0, atol=1e-8)
    result = {
        "ll": float(log_z.sum()),
        "duration_counts": expected,
        "occupancy": occupancy,
        "sum_x": first,
        "sum_x2": second,
    }
    if return_filter:
        result["filtered_state_probabilities"] = alpha.sum(3)
    return result


def statistics(x, pmf, means, variances, *, chunk=16):
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("positive integer chunk required")
    scores(x, means, variances)  # Validate full input before any accumulation.
    result = None
    for start in range(0, len(x), chunk):
        part = chunk_statistics(x[start : start + chunk], pmf, means, variances)
        if result is None:
            result = part
        else:
            for key in result:
                result[key] += part[key]
    return result


def emission_gradients(stats, means, variances):
    n, s, q = stats["occupancy"], stats["sum_x"], stats["sum_x2"]
    means, variances = np.asarray(means), np.asarray(variances)
    return (s - n * means) / variances, (
        -n / variances + (q - 2 * means * s + n * means**2) / variances**2
    ) / 2


def joint_mstep(stats, *, width=8, variance_floor=1e-6):
    n = stats["occupancy"]
    if (
        not np.isfinite(variance_floor)
        or variance_floor <= 0
        or not np.isfinite(n).all()
        or (n <= 0).any()
    ):
        raise ValueError("positive occupancy and variance floor required")
    means = stats["sum_x"] / n
    variances = np.maximum(stats["sum_x2"] / n - means**2, variance_floor)
    if not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise FloatingPointError("nonfinite Gaussian update")
    p, certificate = smooth_mstep(stats["duration_counts"], 0.0, width=width)
    return p, means, variances, certificate


def initialize_from_observations(x, *, iterations=20, variance_floor=1e-6):
    """Deterministic1D Lloyd initialization; training observations only."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
        or not np.isfinite(variance_floor)
        or variance_floor <= 0
    ):
        raise ValueError("positive initialization iteration budget and variance floor required")
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size < 4 or not np.isfinite(values).all():
        raise ValueError("at least four finite training observations required")
    means = np.quantile(values, [0.125, 0.375, 0.625, 0.875])
    for _ in range(iterations):
        assigned = np.abs(values[:, None] - means).argmin(1)
        updated = np.array(
            [values[assigned == k].mean() if (assigned == k).any() else means[k] for k in range(4)]
        )
        if np.max(np.abs(updated - means)) <= 1e-10:
            means = updated
            break
        means = updated
    assigned = np.abs(values[:, None] - means).argmin(1)
    global_var = max(float(values.var()), variance_floor)
    variances = np.array(
        [
            (
                max(float(values[assigned == k].var()), variance_floor)
                if (assigned == k).sum() >= 2
                else global_var
            )
            for k in range(4)
        ]
    )
    order = np.argsort(means)
    return means[order], variances[order]
