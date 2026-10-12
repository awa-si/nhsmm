"""Observation-only production marginal duration learning; known nuisance parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.verify_isolated_trading_hsmm import (
    D,
    MEANS,
    check_recursion,
    make_model,
    model_law,
    segment_reference,
    sequences,
    true_law,
)

SIGMA = 2.0
UPDATES = 80
TRAIN_COUNT = 128
STEPS = 300
BATCH = 2


def independent_ll(x, pmf, sigma=SIGMA):
    total = 0.0
    for sequence in x:
        scores = -0.5 * (
            math.log(2 * math.pi * sigma**2)
            + (sequence[:, 0, None].astype(float) - MEANS) ** 2 / sigma**2
        )
        ll, _ = segment_reference(scores, pmf, BASE_TRANSITION, np.full(4, 0.25))
        total += ll
    return total


def gradient_check(model, x, epsilon=1e-4):
    """Central differences through independent segment recursion, not autograd."""
    logits = model.dist.duration.logits
    logits.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model.log_likelihood(torch.tensor(x), reduce=True).backward()
    gradient = logits.grad.detach().numpy().copy()
    if not np.isfinite(gradient).all():
        raise AssertionError("non-finite public marginal duration gradient")
    pmf = model_law(model).detach().numpy()
    support = pmf > 0
    if np.any(gradient[~support] != 0):
        raise AssertionError("masked-out duration received gradient")
    errors = []
    coordinates = []
    for k in range(4):
        indices = np.flatnonzero(support[k])
        for d in indices[[0, len(indices) // 2, -1]]:
            perturbed = []
            for sign in (1, -1):
                p = pmf.copy()
                p[k, d] *= math.exp(sign * epsilon)
                p[k] /= p[k].sum()
                perturbed.append(independent_ll(x, p))
            finite_difference = (perturbed[0] - perturbed[1]) / (2 * epsilon)
            error = abs(gradient[k, d] - finite_difference)
            errors.append(error)
            coordinates.append(
                {
                    "state": k,
                    "duration": int(d + 1),
                    "autograd": float(gradient[k, d]),
                    "finite_difference": finite_difference,
                    "absolute_error": error,
                }
            )
    error = max(errors)
    if error > 1e-5:
        raise AssertionError(f"independent duration gradient mismatch {error}")
    return {
        "maximum_absolute_error": error,
        "coordinates": coordinates,
        "epsilon": epsilon,
        "masked_gradient_exactly_zero": True,
    }


def marginal_ll(model, x, batch=8):
    with torch.inference_mode():
        return sum(
            float(model.log_likelihood(torch.tensor(x[i : i + batch]), reduce=True))
            for i in range(0, len(x), batch)
        )


def fit_observations(x, *, iterations=UPDATES, seed=901):
    """Labels cannot enter this API: loss is public log p(x), nuisance is frozen."""
    model = make_model(true_law(), uniform=True, sigma=SIGMA)
    logits = model.dist.duration.logits
    logits.requires_grad_(True)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam([logits], lr=0.08)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    pointer = 0
    history = []
    used = set()
    start = time.monotonic()
    for step in range(iterations):
        if pointer + BATCH > len(order):
            order = rng.permutation(len(x))
            pointer = 0
        indices = order[pointer : pointer + BATCH]
        pointer += BATCH
        used.update(int(i) for i in indices)
        optimizer.zero_grad(set_to_none=True)
        loss = -model.log_likelihood(torch.tensor(x[indices]), reduce=True) / (
            len(indices) * x.shape[1]
        )
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all():
            raise AssertionError(f"non-finite loss/gradient at update {step + 1}")
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == iterations:
            history.append(
                {
                    "completed_updates": step + 1,
                    "pre_update_minibatch_nll_per_row": float(loss.detach()),
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            print(f"seed={seed} updates={step + 1}/{iterations}", flush=True)
    for n, p in model.named_parameters():
        if n in frozen and not torch.equal(p.detach(), frozen[n]):
            raise AssertionError(f"frozen nuisance parameter changed: {n}")
    return model, {
        "actual_updates": iterations,
        "unique_training_sequences_seen": len(used),
        "checkpoints": history,
        "frozen_nuisance_unchanged": True,
    }


def exposure(labels):
    """Evaluation only, called after fitting; terminal observed run is censored."""
    complete = np.zeros((4, D), dtype=int)
    censored = np.zeros_like(complete)
    for row in labels:
        starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
        ends = np.r_[starts[1:], len(row)]
        for i, (a, b) in enumerate(zip(starts, ends)):
            target = censored if i == len(starts) - 1 else complete
            target[row[a], b - a - 1] += 1
    return {
        "complete_episodes_by_state": complete.sum(1).tolist(),
        "terminal_censored_episodes_by_state": censored.sum(1).tolist(),
        "supported_bins_with_zero_complete_observations": ((complete == 0) & (true_law() > 0))
        .sum(1)
        .tolist(),
    }


def run(seed):
    truth = true_law()
    x, evaluation_labels = sequences(seed + 400000, count=TRAIN_COUNT, steps=STEPS, sigma=SIGMA)
    heldout, heldout_labels = sequences(seed + 500000, count=24, steps=STEPS, sigma=SIGMA)
    initial = make_model(truth, uniform=True, sigma=SIGMA)
    gradient = gradient_check(initial, x[:1, :300])
    initial_pmf = model_law(initial).detach().numpy()
    before_train = marginal_ll(initial, x)
    before_test = marginal_ll(initial, heldout)
    model, optimization = fit_observations(x, seed=seed)
    fitted = model_law(model).detach().numpy()
    after_train = marginal_ll(model, x)
    after_test = marginal_ll(model, heldout)
    recursion = check_recursion(model, heldout[:2], heldout_labels[:2], sigma=SIGMA)
    ages = np.arange(1, D + 1)
    true_mean = truth @ ages
    initial_mean = initial_pmf @ ages
    learned_mean = fitted @ ages
    train_improvement = (after_train - before_train) / (TRAIN_COUNT * STEPS)
    heldout_improvement = (after_test - before_test) / heldout_labels.size
    initial_rmse = float(np.sqrt(np.mean((initial_mean - true_mean) ** 2)))
    learned_rmse = float(np.sqrt(np.mean((learned_mean - true_mean) ** 2)))
    tv = np.abs(fitted - truth).sum(1) / 2
    return {
        "seed": seed,
        "training_observation_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "heldout_observation_sha256": hashlib.sha256(heldout.tobytes()).hexdigest(),
        "train_rows": int(evaluation_labels.size),
        "heldout_rows": int(heldout_labels.size),
        "training_labels_used": False,
        "gradient_check": gradient,
        "optimization": optimization,
        "train_ll_initial": before_train,
        "train_ll_final": after_train,
        "heldout_ll_initial": before_test,
        "heldout_ll_final": after_test,
        "train_improvement_nats_per_row": train_improvement,
        "heldout_improvement_nats_per_row": heldout_improvement,
        "true_mean": true_mean.tolist(),
        "initial_mean": initial_mean.tolist(),
        "learned_mean": learned_mean.tolist(),
        "initial_mean_rmse": initial_rmse,
        "learned_mean_rmse": learned_rmse,
        "initial_pmf_tv": (np.abs(initial_pmf - truth).sum(1) / 2).tolist(),
        "learned_pmf_tv": tv.tolist(),
        "fitted_pmf": fitted.tolist(),
        "evaluation_only_train_exposure": exposure(evaluation_labels),
        "heldout_recursion": recursion,
        "learning_progress_gate": bool(
            train_improvement > 0 and heldout_improvement > 0 and learned_rmse < initial_rmse
        ),
        "step12_fine_pmf_tv_gate_passed": bool(np.max(tv) <= 0.12),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [
        "scripts/verify_unlabelled_trading_hsmm.py",
        "scripts/verify_isolated_trading_hsmm.py",
        "scripts/diagnose_trading_ground_truth.py",
        "scripts/generate_trading_regime_dataset.py",
        "nhsmm/models/base.py",
        "nhsmm/filtering.py",
        "nhsmm/distributions/default.py",
    ]
    result = {
        "schema": "nhsmm-unlabelled-trading-hsmm-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        },
        "contract": {
            "training": "observations only; public production marginal log_likelihood; Adam lr0.08, 80 updates, batch2; only duration logits trainable",
            "nuisance": "known IID Gaussian means -6/-2/2/6 sigma2.0; known stationary transition with zero diagonal; uniform initial state; fresh age1 starts; exact finite D260 support mask; no tail or same-state resets",
            "data": "128 train and24 independent heldout sequences of300 rows/seed; evaluation labels excluded from fit API and gradient/objective checks",
            "budget_revision": "600-row preliminary execution stopped after last reported40/80 updates for seed901 due backward-pass cost; no final fitted scoring or results retained; sequence length reduced to300 before final three-seed execution, keeping D260, 80 updates and acceptance criteria",
            "verification": "public autograd versus independent segment central differences on12 coordinates <=1e-5; finite gradients throughout; frozen nuisance equality; heldout LL<=1e-7 and state probability<=1e-8 reference error",
            "acceptance": "report separately: positive train and heldout LL improvement plus reduced duration-mean RMSE; stricter each-PMF-TV<=0.12 recovery criterion from step12",
            "limits": "bounded stochastic duration-only learning, not stock optimize or joint nuisance learning; known support/nuisance and fresh starts; not Trading-feature or macro-context recovery; finite-sample exposure may not identify every PMF bin; masks remain script-only",
        },
        "runs": [run(seed) for seed in (901, 902, 903)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "seed",
                        "train_improvement_nats_per_row",
                        "heldout_improvement_nats_per_row",
                        "learned_mean_rmse",
                        "learned_pmf_tv",
                        "learning_progress_gate",
                        "step12_fine_pmf_tv_gate_passed",
                    )
                }
                for r in result["runs"]
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
