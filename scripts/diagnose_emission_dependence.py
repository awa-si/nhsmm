"""Controlled covariance and AR edge-emission filtering; no production changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from scripts.diagnose_model_conform_duration import PMF, TAIL, TRANSITION, generate

MEANS = np.array([[-0.75, -0.75], [0.75, 0.75]])
SCENARIOS = {
    "iid_diagonal": (0.0, 0.0),
    "iid_correlated": (0.0, 0.75),
    "ar_diagonal": (0.8, 0.0),
    "ar_correlated": (0.8, 0.75),
}
ARMS = ("diagonal_iid", "full_iid", "diagonal_ar", "full_ar")


def observations(labels, seed, rho, correlation):
    """Stationary residual AR independent of regime; sequence/prefix-stable noise."""
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    chol = np.linalg.cholesky(covariance)
    rows = []
    for child, states in zip(np.random.SeedSequence(seed).spawn(len(labels)), labels):
        innovations = np.random.default_rng(child).normal(size=(len(states), 2)) @ chol.T
        residual = innovations.copy()
        for t in range(1, len(states)):
            residual[t] = rho * residual[t - 1] + math.sqrt(1 - rho**2) * innovations[t]
        rows.append(MEANS[states] + residual)
    return np.stack(rows)


def fit_emissions(x, labels):
    """Privileged train-only nuisance fit, not unsupervised representation learning."""
    means = np.stack([x[labels == k].mean(0) for k in range(2)])
    residual = x - means[labels]
    covariance = np.einsum("btf,btg->fg", residual, residual) / labels.size
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
    previous, current = residual[:, :-1], residual[:, 1:]
    rho = float(np.clip(np.sum(previous * current) / np.sum(previous**2), -0.95, 0.95))
    return {"means": means, "covariance": covariance, "rho": rho}


def gaussian_log_density(value, mean, covariance):
    difference = value - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("positive definite covariance required")
    quadratic = np.einsum("...f,fg,...g->...", difference, np.linalg.inv(covariance), difference)
    return -0.5 * (2 * math.log(2 * math.pi) + logdet + quadratic)


def edge_log_density(current, previous, means, covariance, rho):
    # Gaussian residual = x_t - mu[z_t]. Its past value depends on z_(t-1).
    conditional_mean = means[None, None] + rho * (previous[:, None, None] - means[None, :, None])
    return gaussian_log_density(current[:, None, None], conditional_mean, covariance * (1 - rho**2))


def filter_observations(x, means, covariance, rho, pmf=PMF, hazard=TAIL, transition=TRANSITION):
    """Scaled probability-space causal age-chain with state-pair emission densities."""
    batch, steps, _ = x.shape
    width = pmf.shape[1]
    end = pmf / np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    end[:, -1] = hazard
    frontier = np.zeros((batch, 2, width))
    frontier[:, :, 0] = 0.5
    posterior, increments = [], []
    for t in range(steps):
        if t == 0:
            log_density = gaussian_log_density(x[:, 0, None], means, covariance)
            offset = log_density.max(1)
            frontier *= np.exp(log_density - offset[:, None])[:, :, None]
        else:
            log_density = edge_log_density(x[:, t], x[:, t - 1], means, covariance, rho)
            offset = log_density.max((1, 2))
            density = np.exp(log_density - offset[:, None, None])
            following = np.zeros_like(frontier)
            following[:, :, 0] += np.einsum("bka,ka,kj,bkj->bj", frontier, end, transition, density)
            same_state = np.stack([density[:, k, k] for k in range(2)], axis=1)
            continuing = frontier * (1 - end)[None] * same_state[:, :, None]
            following[:, :, 1:] += continuing[:, :, :-1]
            following[:, :, -1] += continuing[:, :, -1]
            frontier = following
        scale = frontier.sum((1, 2))
        if not np.all(scale > 0):
            raise ValueError("zero filtering evidence")
        frontier /= scale[:, None, None]
        posterior.append(frontier.sum(2))
        increments.append(np.log(scale) + offset)
    return np.stack(posterior, axis=1), np.stack(increments, axis=1)


def score(x, labels, parameters, arm):
    covariance = parameters["covariance"]
    if arm.startswith("diagonal"):
        covariance = np.diag(np.diag(covariance))
    rho = parameters["rho"] if arm.endswith("ar") else 0.0
    posterior, increments = filter_observations(x, parameters["means"], covariance, rho)
    onehot = np.eye(2)[labels]
    probability_true = np.take_along_axis(posterior, labels[:, :, None], axis=2).squeeze(2)
    predicted = posterior.argmax(2)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "state_log_score": float(np.log(np.maximum(probability_true, 1e-300)).mean()),
        "brier": float(((posterior - onehot) ** 2).sum(2).mean()),
        "test_ll_per_row": float(increments.mean()),
        "predicted_switches": int(np.sum(predicted[:, 1:] != predicted[:, :-1])),
        "true_switches": int(np.sum(labels[:, 1:] != labels[:, :-1])),
        "rho": rho,
        "covariance": covariance.tolist(),
    }


def run(seed):
    _, train_labels, _ = generate(seed + 20000, tail=True, count=128)
    _, test_labels, _ = generate(seed + 30000, tail=True, count=32)
    rows = {}
    for name, (rho, correlation) in SCENARIOS.items():
        train = observations(train_labels, seed + 40000, rho, correlation)
        test = observations(test_labels, seed + 50000, rho, correlation)
        parameters = fit_emissions(train, train_labels)
        arms = {arm: score(test, test_labels, parameters, arm) for arm in ARMS}
        rows[name] = {
            "true_rho": rho,
            "true_feature_correlation": correlation,
            "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
            "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
            "train_means": parameters["means"].tolist(),
            "train_covariance": parameters["covariance"].tolist(),
            "train_rho": parameters["rho"],
            "arms": arms,
        }
        print(f"seed={seed} scenario={name} complete", flush=True)
    return {
        "seed": seed,
        "train_labels_sha256": hashlib.sha256(train_labels.tobytes()).hexdigest(),
        "test_labels_sha256": hashlib.sha256(test_labels.tobytes()).hexdigest(),
        "scenarios": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    runs = [run(seed) for seed in (901, 902, 903)]
    metrics = ("accuracy", "brier", "state_log_score", "test_ll_per_row")
    summary = {
        scenario: {
            arm: {
                m: float(np.median([r["scenarios"][scenario]["arms"][arm][m] for r in runs]))
                for m in metrics
            }
            for arm in ARMS
        }
        for scenario in SCENARIOS
    }
    payload = {
        "schema": "nhsmm-emission-dependence-diagnostic-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "workers": 1,
            "source_sha256": {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [__file__, "scripts/diagnose_model_conform_duration.py"]
            },
        },
        "generative_law": {
            "means": MEANS.tolist(),
            "pmf": PMF.tolist(),
            "tail_hazard": TAIL.tolist(),
            "transition": TRANSITION.tolist(),
            "initial": [0.5, 0.5],
        },
        "contract": {
            "train_sequences": 128,
            "test_sequences": 32,
            "steps": 128,
            "seeds": [901, 902, 903],
            "observations": "same latent labels and base Gaussian draws across scenarios; stationary unit per-feature residual variance; continuous AR residual across latent boundaries",
            "nuisance_fit": "train labels estimate regime means, shared marginal covariance and pooled residual rho; no test labels enter filtering",
            "regularization": "covariance eigenvalues floor 1e-6; fitted abs(rho)<=.95; no model or scenario selection",
            "inference": "known duration/transition/initial law; causal state-pair AR emissions and age-chain recursion; no post-terminal boundary",
            "limits": "privileged supervised nuisance control and synthetic prototype, not public full-covariance/AR support, unsupervised fitting or real-market acceptance",
        },
        "runs": runs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
