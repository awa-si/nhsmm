"""Replicate Trading heads with direct observable baselines and raw state logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import filter_model_sequence
from scripts.diagnose_model_conform_duration import optimization_evidence
from scripts.diagnose_trading_feature_contract import (
    MatchedFamilyModel,
    WARMUP,
    fit_return_head,
    return_scores,
)
from scripts.generate_trading_regime_dataset import generate
from scripts.validate_trading_regime_usefulness import (
    FEATURES,
    RunBudget,
    _best_perm,
    _config,
    _standardize,
    _standardize_context,
    _windows,
)

LAGS = (0, 1, 3, 12, 48)
RIDGE = 1e-3
FIT_SEED = 901


def observable_design(features, contexts, posterior=None, *, lags=LAGS):
    """Inputs at t and fixed earlier rows; targets are strictly at t+1."""
    if min(lags) < 0 or max(lags) > WARMUP:
        raise ValueError("lags must be causal and fit within the warmup")
    observed = np.concatenate([features, contexts], axis=-1).astype(np.float64)
    end = observed.shape[1] - 1
    pieces = [np.ones((len(observed), end - WARMUP, 1))]
    pieces.extend(observed[:, WARMUP - lag : end - lag] for lag in lags)
    if posterior is not None:
        pieces.append(np.asarray(posterior[:, WARMUP:end], dtype=np.float64))
    joined = np.concatenate(pieces, axis=-1)
    return joined.reshape(-1, joined.shape[-1])


def ridge_fit(design, target):
    penalty = np.diag([0.0] + [RIDGE] * (design.shape[1] - 1))
    return np.linalg.solve(
        design.T @ design / len(target) + penalty, design.T @ target / len(target)
    )


def fit_head(design, returns, *, heteroscedastic):
    """Train-only ridge mean and optional log-residual-variance calibration."""
    target = returns[:, WARMUP + 1 :].reshape(-1).astype(np.float64)
    mean, scale = float(target.mean()), max(float(target.std()), 1e-8)
    normalized = (target - mean) / scale
    beta = ridge_fit(design, normalized)
    squared = (normalized - design @ beta) ** 2
    if heteroscedastic:
        gamma = ridge_fit(design, np.log(np.maximum(squared, 1e-6)))
        preliminary = np.exp(np.clip(design @ gamma, -12, 12))
        smear = max(float(np.mean(squared / preliminary)), 1e-6)
    else:
        gamma = np.zeros(design.shape[1])
        smear = max(float(squared.mean()), 1e-6)
    return {"mean": mean, "scale": scale, "beta": beta, "gamma": gamma, "smear": smear}


def score_head(design, returns, head):
    target = returns[:, WARMUP + 1 :].reshape(-1).astype(np.float64)
    prediction = design @ head["beta"]
    variance = np.maximum(head["smear"] * np.exp(np.clip(design @ head["gamma"], -12, 12)), 1e-6)
    residual = (target - head["mean"]) / head["scale"] - prediction
    scores = -0.5 * (np.log(2 * math.pi * variance) + residual**2 / variance)
    scores -= math.log(head["scale"])
    error = residual * head["scale"]
    return {
        "next_return_log_score": float(scores.mean()),
        "next_return_rmse": float(np.sqrt(np.mean(error**2))),
        "pairs": int(target.size),
    }


def raw_state_scores(log_probability, labels, permutation):
    """Gather truth in log space before any exponentiation; no metric floor."""
    aligned = np.asarray(log_probability[:, WARMUP:], dtype=np.float64)[
        ..., np.argsort(permutation)
    ]
    truth = labels[:, WARMUP:]
    chosen = np.take_along_axis(aligned, truth[..., None], axis=-1).squeeze(-1)
    if not np.isfinite(chosen).all():
        raise ValueError("nonfinite raw true-state log posterior")
    probability = np.exp(aligned)
    return {
        "accuracy": float(np.mean(aligned.argmax(-1) == truth)),
        "brier": float(((probability - np.eye(4)[truth]) ** 2).sum(-1).mean()),
        "state_log_score": float(chosen.mean()),
        "minimum_true_state_log_probability": float(chosen.min()),
        "float32_exp_zero_true_probabilities": int(np.sum(np.exp(chosen.astype(np.float32)) == 0)),
        "rows": int(truth.size),
    }


def serial_head(head):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in head.items()}


def calibrated_result(train_design, test_design, train_returns, test_returns, *, hetero):
    head = fit_head(train_design, train_returns, heteroscedastic=hetero)
    return {
        "return": score_head(test_design, test_returns, head),
        "head": serial_head(head),
        "design_columns": int(train_design.shape[1]),
    }


def run(generator_seed):
    torch.set_num_threads(1)
    data, metadata = generate(
        rows=60000,
        seed=generator_seed,
        start_price=50000.0,
        transition_context_mode="episode_end",
    )
    prior_path = Path("docs/validation/results/trading-residual-step6-native-20261012.json")
    prior_bytes = prior_path.read_bytes()
    prior = next(
        r for r in json.loads(prior_bytes)["runs"] if r["generator_seed"] == generator_seed
    )
    raw = np.column_stack([data[f] for f in FEATURES]).astype(float)
    if hashlib.sha256(raw.tobytes()).hexdigest() != prior["feature_sha256"]:
        raise AssertionError("frozen Step 6 dataset mismatch")
    train = _windows(data, 0, 36000, 8, window=240)
    test = _windows(data, 48000, 60000, 4, window=240)
    tx, vx = _standardize(train[0], test[0])
    tc, vc = _standardize_context(train[1], test[1])
    train_returns, test_returns = train[0][:, :, 0], test[0][:, :, 0]
    baselines = {}
    for name, lags, hetero in (
        ("instant_homoscedastic", (0,), False),
        ("instant_heteroscedastic", (0,), True),
        ("lagged_homoscedastic", LAGS, False),
        ("lagged_heteroscedastic", LAGS, True),
    ):
        baselines[name] = calibrated_result(
            observable_design(tx, tc, lags=lags),
            observable_design(vx, vc, lags=lags),
            train_returns,
            test_returns,
            hetero=hetero,
        )
    baselines["current_return_only"] = {
        "return": return_scores(test_returns, None, fit_return_head(train_returns, None))
    }
    budget = RunBudget(
        window=240, train_windows=8, eval_windows=4, max_iter=40, transition_refine_steps=0
    )
    arms = {}
    for family in ("gaussian", "studentt"):
        np.random.seed(FIT_SEED)
        random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        cfg = _config(
            FIT_SEED,
            context=True,
            hmm=False,
            emission_init_mode="kmeans" if family == "studentt" else "kmeans_cluster_variance",
            budget=budget,
        ).with_overrides(n_features=len(FEATURES), emission_type=family)
        model = MatchedFamilyModel(cfg, device="cpu")
        model.initialize_distributions()
        model.optimize(torch.tensor(tx), context=torch.tensor(tc))
        model.eval()
        evidence = optimization_evidence(model, 40)
        with torch.inference_mode():
            train_log = torch.logsumexp(
                filter_model_sequence(
                    model, torch.tensor(tx), context=torch.tensor(tc)
                ).log_posterior,
                dim=-1,
            ).numpy()
            test_log = torch.logsumexp(
                filter_model_sequence(
                    model, torch.tensor(vx), context=torch.tensor(vc)
                ).log_posterior,
                dim=-1,
            ).numpy()
        train_p, test_p = np.exp(train_log), np.exp(test_log)
        permutation, train_accuracy = _best_perm(
            train[2][:, WARMUP:], train_log[:, WARMUP:].argmax(-1)
        )
        augmented = calibrated_result(
            observable_design(tx, tc, train_p),
            observable_design(vx, vc, test_p),
            train_returns,
            test_returns,
            hetero=True,
        )
        arms[family] = {
            "state": raw_state_scores(test_log, test[2], permutation),
            "train_permutation": list(permutation),
            "train_mapping_accuracy": train_accuracy,
            "optimization": evidence,
            "step7_return_head": {
                "return": return_scores(
                    test_returns, test_p, fit_return_head(train_returns, train_p)
                )
            },
            "lagged_plus_posterior_heteroscedastic": augmented,
            "increment_vs_lagged_heteroscedastic": {
                k: augmented["return"][k] - baselines["lagged_heteroscedastic"]["return"][k]
                for k in ("next_return_log_score", "next_return_rmse")
            },
        }
        print(f"dataset={generator_seed} family={family} complete", flush=True)
    return {
        "generator_seed": generator_seed,
        "fit_seed": FIT_SEED,
        "dataset_schema": metadata["schema"],
        "dataset_feature_sha256": prior["feature_sha256"],
        "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "baselines": baselines,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-seed", type=int, choices=[424242, 424243, 424244], required=True
    )
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        "scripts/diagnose_trading_information_baselines.py",
        "scripts/diagnose_trading_feature_contract.py",
        "scripts/diagnose_model_conform_duration.py",
        "scripts/generate_trading_regime_dataset.py",
        "scripts/validate_trading_regime_usefulness.py",
    ]
    payload = {
        "schema": "nhsmm-trading-information-baselines-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        },
        "contract": {
            "dataset_seeds": [424242, 424243, 424244],
            "fit_seed": FIT_SEED,
            "features": list(FEATURES),
            "context": "macro, train-only standardized",
            "train_range": [0, 36000],
            "test_range": [48000, 60000],
            "windows": "8 train/4 test windows of 240; fixed 64-row warmup, 704 state rows/700 future pairs",
            "model": "K4 D64 plus tail, encoderless external context, one restart, 40 updates, fixed temperature1, no transition refinement",
            "direct_baselines": "All 8 features plus macro, current or fixed causal lags; same raw next-return targets",
            "lags": list(LAGS),
            "ridge": RIDGE,
            "heads": "train-only ridge mean; pooled or ridge log squared training residuals with train-only multiplicative smearing; log variance clip [-12,12], normalized variance floor1e-6",
            "augmentation": "identical lagged heteroscedastic head plus four frozen current HSMM posterior columns",
            "labels": "only train permutation for state evaluation; no state labels in model or forecast heads",
            "raw_state_logs": "logsumexp of state-age log posterior before any exp; no log metric floor",
            "limits": "synthetic sparse-window pilot, fixed single fit seed; HSMM uses entire window prefix while direct lags reach48; in-sample residual variance calibration and fixed ridge may overfit; no hyperparameter selection, convergence, significance, walk-forward or trading-utility claim",
        },
        "run": run(args.generator_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "baselines": {k: v["return"] for k, v in payload["run"]["baselines"].items()},
                "arms": {
                    k: {"state": v["state"], "increment": v["increment_vs_lagged_heteroscedastic"]}
                    for k, v in payload["run"]["arms"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
