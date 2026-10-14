"""Research-only concave fixed-bin log-density smoothing M-step."""

from __future__ import annotations

import numpy as np
import torch

from scripts.compare_adaptive_duration_em import PSEUDOCOUNT


def geometry(size, width):
    """Full-age bins and spacing-aware log-density curvature operator."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (size, width)):
        raise ValueError("positive integer size/width required")
    # The older compact helper keeps endpoint atoms free; this protocol uses
    # a different fixed grid with no special endpoint singleton bins.
    blocks = [(start, min(start + width, size)) for start in range(0, size, width)]
    lengths = np.array([b - a for a, b in blocks], dtype=int)
    centers = np.array([(a + b - 1) / 2 for a, b in blocks], dtype=float)
    gaps = np.diff(centers) / width
    r = np.zeros((max(0, len(blocks) - 2), len(blocks)))
    for i in range(len(r)):
        r[i, i] = 1 / gaps[i]
        r[i, i + 1] = -(1 / gaps[i] + 1 / gaps[i + 1])
        r[i, i + 2] = 1 / gaps[i + 1]
    return blocks, lengths, r


def count_objective(z, masses, lengths, curvature, strength):
    """Concave raw augmented complete-data objective, one value per state."""
    log_norm = torch.logsumexp(z + lengths.log(), dim=1)
    roughness = (z @ curvature.T).square().sum(1)
    return (masses * z).sum(1) - masses.sum(1) * log_norm - strength * roughness / 2


def smooth_mstep(
    counts,
    strength,
    *,
    width=8,
    pseudocount=PSEUDOCOUNT,
    initial_pmf=None,
    max_iterations=1000,
    max_evaluations=2000,
    gradient_tolerance=1e-8,
):
    """Solve in float64; fail closed on stationarity/Q/PMF violations.

    The additive shift is removed by anchoring the last bin's log density to0.
    Lambda0 is analytic. Positive lambda uses the declared Torch optimizer;
    normalized anchored-gradient residuals certify numerical stationarity,
    not a statistical recovery rate or an exact objective-gap bound.
    """
    counts = np.asarray(counts, dtype=float)
    if (
        counts.ndim != 2
        or not counts.size
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.isfinite(strength)
        or strength < 0
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
        or not np.isfinite(gradient_tolerance)
        or gradient_tolerance <= 0
        or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in (max_iterations, max_evaluations)
        )
    ):
        raise ValueError("finite nonnegative counts/strength and positive solver controls required")
    blocks, lengths, r = geometry(counts.shape[1], width)
    starts = np.array([a for a, b in blocks])
    augmented = np.add.reduceat(counts + pseudocount, starts, axis=1)
    analytic = augmented / augmented.sum(1, keepdims=True) / lengths
    initial = (
        np.repeat(analytic, lengths, axis=1)
        if initial_pmf is None
        else np.array(initial_pmf, dtype=float, copy=True)
    )
    if (
        initial.shape != counts.shape
        or not np.isfinite(initial).all()
        or np.any(initial <= 0)
        or not np.allclose(initial.sum(1), 1, rtol=0, atol=1e-12)
        or not np.allclose(
            np.repeat(initial[:, starts], lengths, axis=1), initial, rtol=0, atol=1e-15
        )
    ):
        raise ValueError("initial PMF must be positive normalized and constant within fixed bins")
    mass = torch.tensor(augmented, dtype=torch.float64)
    length = torch.tensor(lengths, dtype=torch.float64)
    operator = torch.tensor(r, dtype=torch.float64)
    initial_z = torch.tensor(np.log(initial[:, starts]), dtype=torch.float64)
    initial_z = initial_z - initial_z[:, -1:]
    before = count_objective(initial_z, mass, length, operator, strength).detach()
    evaluations = 0
    iterations = 0
    if strength == 0 or len(blocks) == 1:
        z = torch.tensor(np.log(analytic), dtype=torch.float64)
        z = z - z[:, -1:]
    else:
        # Counts differ greatly across states in a censored short episode.
        # Whiten with each state's initial normalized negative Hessian instead
        # of asking one line search to handle incompatible curvature scales.
        initial_mass = torch.softmax(initial_z + length.log(), dim=1)
        hessian = torch.diag_embed(initial_mass) - initial_mass.unsqueeze(
            2
        ) * initial_mass.unsqueeze(1)
        hessian = hessian + (strength / mass.sum(1))[:, None, None] * (operator.T @ operator)
        factor = torch.linalg.cholesky(hessian[:, :-1, :-1])
        free = torch.zeros((len(counts), len(blocks) - 1), dtype=torch.float64, requires_grad=True)

        def expanded():
            delta = torch.linalg.solve_triangular(
                factor.transpose(1, 2), free.unsqueeze(2), upper=True
            ).squeeze(2)
            return torch.cat((initial_z[:, :-1] + delta, torch.zeros_like(delta[:, :1])), dim=1)

        optimizer = torch.optim.LBFGS(
            [free],
            max_iter=max_iterations,
            max_eval=max_evaluations,
            tolerance_grad=1e-13,
            tolerance_change=0.0,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal evaluations
            evaluations += 1
            if evaluations > max_evaluations:
                raise FloatingPointError("smoothing solver evaluation budget exhausted")
            optimizer.zero_grad()
            z = expanded()
            loss = -(count_objective(z, mass, length, operator, strength) / mass.sum(1)).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        iterations = optimizer.state[free]["n_iter"]
        z = expanded().detach()
        # Wolfe searches lose objective differences near machine precision.
        # A few native Newton corrections polish the original gradient, using
        # the independently tested Hessian; never expand the iteration budget.
        for _ in range(min(3, max_iterations - iterations)):
            bin_mass = torch.softmax(z + length.log(), dim=1)
            scale = strength / mass.sum(1)
            grad = (
                mass / mass.sum(1, keepdim=True)
                - bin_mass
                - scale[:, None] * (z @ operator.T @ operator)
            )
            if float(grad[:, :-1].abs().max()) <= gradient_tolerance:
                break
            curvature = torch.diag_embed(bin_mass) - bin_mass.unsqueeze(2) * bin_mass.unsqueeze(1)
            curvature = curvature + scale[:, None, None] * (operator.T @ operator)
            delta = torch.linalg.solve(curvature[:, :-1, :-1], grad[:, :-1].unsqueeze(2)).squeeze(2)
            candidate = torch.cat((z[:, :-1] + delta, torch.zeros_like(delta[:, :1])), dim=1)
            old_q = count_objective(z, mass, length, operator, strength)
            new_q = count_objective(candidate, mass, length, operator, strength)
            if not torch.isfinite(new_q).all() or torch.any(new_q < old_q - 1e-7):
                break
            z = candidate
            iterations += 1
    z = z.detach().requires_grad_(True)
    after = count_objective(z, mass, length, operator, strength)
    gradient = torch.autograd.grad((after / mass.sum(1)).sum(), z)[0]
    residual = float(gradient[:, :-1].abs().max()) if len(blocks) > 1 else 0.0
    log_density = z - torch.logsumexp(z + length.log(), dim=1, keepdim=True)
    p = torch.repeat_interleave(log_density.exp(), torch.tensor(lengths), dim=1).detach().numpy()
    if (
        not np.isfinite(p).all()
        or np.any(p <= 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12)
        or not np.isfinite(residual)
        or residual > gradient_tolerance
        or torch.any(after.detach() < before - 1e-7)
    ):
        raise FloatingPointError(
            f"smoothing M-step acceptance failed: normalized gradient={residual}, "
            f"iterations={iterations}, evaluations={evaluations}, "
            f"state_residuals={gradient[:, :-1].abs().max(1).values.tolist()}"
        )
    return p, {
        "maximum_normalized_anchored_gradient": residual,
        "q_start": before.tolist(),
        "q_final": after.detach().tolist(),
        "roughness": (z.detach() @ operator.T).square().sum(1).tolist(),
        "solver_iterations": iterations,
        "closure_evaluations": evaluations,
    }


def marginal_objective(ll, p, strength, width=8, pseudocount=PSEUDOCOUNT):
    p = np.asarray(p, dtype=float)
    if (
        p.ndim != 2
        or not p.size
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12)
        or not np.isfinite(ll)
        or not np.isfinite(strength)
        or strength < 0
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise ValueError("finite LL/strength and positive normalized PMF/pseudocount required")
    blocks, _, r = geometry(p.shape[1], width)
    z = np.log(p[:, [a for a, b in blocks]])
    lengths = [b - a for a, b in blocks]
    if not np.allclose(np.repeat(np.exp(z), lengths, axis=1), p, rtol=0, atol=1e-15):
        raise ValueError("PMF must be constant within fixed bins")
    return float(ll + pseudocount * np.log(p).sum() - strength * np.square(z @ r.T).sum() / 2)
