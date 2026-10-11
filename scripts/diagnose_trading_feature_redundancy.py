"""Predeclared feature redundancy pilot with proper Gaussian likelihoods."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nhsmm import NHSMM
from nhsmm.filtering import filter_model_sequence
from scripts.diagnose_trading_decoder_duration import score
from scripts.validate_trading_regime_usefulness import (
    FEATURES,
    RunBudget,
    _config,
    _ll,
    _standardize,
    _standardize_context,
    _windows,
)

REDUCED = ("log_return_1", "trend_48", "ewma_vol_12", "volume_z_48")
ARMS = ("full8", "reduced4", "whiten8")


@dataclass(frozen=True)
class Whitening:
    mean: np.ndarray
    matrix: np.ndarray
    log_abs_det: float
    eigenvalues: np.ndarray

    def apply(self, observations):
        return ((observations - self.mean) @ self.matrix).astype(np.float32)


def fit_whitening(train, eigenvalue_floor=1e-3):
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive and finite")
    flat = np.asarray(train, dtype=np.float64).reshape(-1, train.shape[-1])
    if len(flat) < 2 or not np.isfinite(flat).all():
        raise ValueError("finite training observations with at least two rows required")
    mean = flat.mean(0)
    centered = flat - mean
    covariance = centered.T @ centered / len(flat)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    matrix = (vectors * (1 / np.sqrt(clipped))) @ vectors.T
    return Whitening(mean, matrix, float(-0.5 * np.log(clipped).sum()), eigenvalues)


def fit(seed, x, c, budget):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = _config(
        seed, context=True, hmm=False, emission_init_mode="kmeans_cluster_variance", budget=budget
    )
    model = NHSMM(cfg.with_overrides(n_features=x.shape[-1]), device="cpu")
    model.initialize_distributions()
    model.optimize(torch.tensor(x), context=torch.tensor(c))
    model.eval()
    return model


def run(job):
    seed, path = job
    torch.set_num_threads(1)
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    n = len(data["close"])
    budget = RunBudget(
        window=240, train_windows=8, eval_windows=4, max_iter=4, transition_refine_steps=2
    )
    train = _windows(data, 0, int(0.6 * n), 8, window=240)
    test = _windows(data, int(0.8 * n), n, 4, window=240)
    train_x, test_x = _standardize(train[0], test[0])
    train_c, test_c = _standardize_context(train[1], test[1])
    whitening = fit_whitening(train_x)
    reduced = [FEATURES.index(name) for name in REDUCED]
    results = []
    for arm in ARMS:
        if arm == "full8":
            tx, vx, features, jacobian = train_x, test_x, list(FEATURES), 0.0
        elif arm == "reduced4":
            tx, vx, features, jacobian = (
                train_x[:, :, reduced],
                test_x[:, :, reduced],
                list(REDUCED),
                None,
            )
        else:
            tx, vx, features, jacobian = (
                whitening.apply(train_x),
                whitening.apply(test_x),
                list(FEATURES),
                whitening.log_abs_det,
            )
        model = fit(seed, tx, train_c, budget)
        with torch.inference_mode():
            posterior = filter_model_sequence(
                model, torch.tensor(vx), context=torch.tensor(test_c)
            ).state_posterior.numpy()
        result = score(test[2], posterior.argmax(-1))
        ll = _ll(model, vx, test_c)
        corr = np.corrcoef(tx.reshape(-1, tx.shape[-1]), rowvar=False)
        result.update(
            arm=arm,
            features=features,
            dimension=tx.shape[-1],
            transformed_ll_per_row=ll,
            log_abs_det=jacobian,
            comparable_full8_ll_per_row=ll + jacobian if jacobian is not None else None,
            max_abs_training_offdiag_correlation=float(
                np.max(np.abs(corr - np.eye(len(features))))
            ),
        )
        results.append(result)
    return {
        "seed": seed,
        "arms": results,
        "true_switches": int(np.count_nonzero(test[2][:, 1:] != test[2][:, :-1])),
        "whitening": {
            "mean": whitening.mean.tolist(),
            "matrix": whitening.matrix.tolist(),
            "eigenvalues": whitening.eigenvalues.tolist(),
            "eigenvalue_floor": 1e-3,
            "log_abs_det": whitening.log_abs_det,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end data required")
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, [(s, args.dataset) for s in (901, 902, 903)]))
    summary = {}
    for arm in ARMS:
        group = [a for row in rows for a in row["arms"] if a["arm"] == arm]
        metrics = (
            "accuracy",
            "ari",
            "boundary_f1_tol5",
            "run_duration_rel_error",
            "switches",
            "transformed_ll_per_row",
            "max_abs_training_offdiag_correlation",
        )
        result = {m: float(np.median([a[m] for a in group])) for m in metrics}
        result["comparable_full8_ll_per_row"] = (
            float(np.median([a["comparable_full8_ll_per_row"] for a in group]))
            if arm != "reduced4"
            else None
        )
        summary[arm] = result
    payload = {
        "schema": "nhsmm-trading-feature-redundancy-pilot-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seeds": [901, 902, 903],
        "window": 240,
        "train_windows": 8,
        "eval_windows": 4,
        "max_iter": 4,
        "transition_refine_steps": 2,
        "full_protocol": False,
        "arms_predeclared": list(ARMS),
        "runs": rows,
        "summary": summary,
        "contract": {
            "labels": "scoring only; no feature/transform fit or model selection",
            "whitening": "pointwise ZCA transform fitted solely to chronological training windows",
            "likelihood": "full8 versus whiten8 includes log-Jacobian in original standardized 8D space",
            "reduced4": "likelihood is only a 4D density; cannot compare with full8 density",
            "covariance": "pooled whitening does not guarantee state-conditional decorrelation",
            "training": "same seeds and budget; dimensions/initial clustering differ across arms",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
