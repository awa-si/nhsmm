"""Train-only oracle residual diagnostics on maintained synthetic Trading features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from scripts.generate_trading_regime_dataset import REGIMES, generate
from scripts.validate_trading_regime_usefulness import FEATURES

LAGS = (1, 3, 12, 48)
RIDGE = 1e-4


def lag_mask(episodes, lag):
    """Monotone episode IDs ensure all intermediate rows share an episode."""
    return episodes[lag:] == episodes[:-lag]


def fit_controls(x, labels, episodes, n_states=4):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    state_mean = np.stack([z[labels == k].mean(0) for k in range(n_states)])
    residual = z - state_mean[labels]
    scalar, matrix = [], []
    pairs = lag_mask(episodes, 1)
    for state in range(n_states):
        mask = pairs & (labels[1:] == state)
        previous, current = residual[:-1][mask], residual[1:][mask]
        if len(current) < x.shape[1] + 2:
            raise ValueError("insufficient within-episode training pairs")
        # Include a fitted intercept; ridge penalizes slope coefficients only.
        ones = np.ones((len(current), 1))
        slopes, intercepts = [], []
        for feature in range(x.shape[1]):
            design = np.column_stack([ones, previous[:, feature]])
            penalty = np.diag([0.0, RIDGE])
            coefficient = np.linalg.solve(
                design.T @ design / len(current) + penalty,
                design.T @ current[:, feature] / len(current),
            )
            intercepts.append(coefficient[0])
            slopes.append(coefficient[1])
        scalar.append({"intercept": np.array(intercepts), "slopes": np.array(slopes)})
        design = np.column_stack([ones, previous])
        penalty = np.diag([0.0] + [RIDGE] * x.shape[1])
        coefficients = np.linalg.solve(
            design.T @ design / len(current) + penalty, design.T @ current / len(current)
        )
        matrix.append(coefficients)
    return {
        "mean": mean,
        "scale": scale,
        "state_mean": state_mean,
        "scalar": scalar,
        "matrix": matrix,
    }


def residual_variants(x, labels, episodes, parameters):
    z = (x - parameters["mean"]) / parameters["scale"]
    residual = z - parameters["state_mean"][labels]
    scalar = np.full_like(residual, np.nan)
    matrix = np.full_like(residual, np.nan)
    pairs = lag_mask(episodes, 1)
    for state in range(len(parameters["state_mean"])):
        selected = np.flatnonzero(pairs & (labels[1:] == state)) + 1
        previous = residual[selected - 1]
        ar = parameters["scalar"][state]
        scalar[selected] = residual[selected] - ar["intercept"] - previous * ar["slopes"]
        design = np.column_stack([np.ones(len(selected)), previous])
        matrix[selected] = residual[selected] - design @ parameters["matrix"][state]
    return {"mean_only": residual, "diagonal_ar1": scalar, "var1": matrix}


def correlation(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    value = np.divide(
        np.sum(a * b, axis=0),
        denominator,
        out=np.full(a.shape[1], np.nan),
        where=denominator > 1e-20,
    )
    return [float(v) if np.isfinite(v) else None for v in value]


def summarize(residual, labels, episodes, state, features):
    valid = np.isfinite(residual).all(1)
    rows = residual[valid & (labels == state)]
    if len(rows) < 2:
        raise ValueError("insufficient state rows for residual diagnostics")
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / len(rows)
    variance = np.diag(covariance)
    denominator = np.sqrt(variance[:, None] * variance[None])
    corr = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-20
    )
    excess = (
        np.divide(
            np.mean(centered**4, axis=0),
            variance**2,
            out=np.full(len(features), np.nan),
            where=variance > 1e-20,
        )
        - 3
    )
    eigenvalues = np.linalg.eigvalsh(corr)
    acf, pair_counts = {}, {}
    for lag in LAGS:
        mask = lag_mask(episodes, lag) & (labels[lag:] == state) & valid[lag:] & valid[:-lag]
        pair_counts[str(lag)] = int(mask.sum())
        acf[str(lag)] = (
            correlation(residual[:-lag][mask], residual[lag:][mask])
            if mask.sum() >= 2
            else [None] * len(features)
        )
    maximum = np.abs(corr - np.diag(np.diag(corr)))
    i, j = np.unravel_index(np.argmax(maximum), maximum.shape)
    return {
        "state": state,
        "regime": REGIMES[state],
        "rows": len(rows),
        "acf": acf,
        "lag_pair_counts": pair_counts,
        "correlation": corr.tolist(),
        "correlation_eigenvalues": eigenvalues.tolist(),
        "max_abs_cross_correlation": float(maximum[i, j]),
        "max_correlation_pair": [features[i], features[j]],
        "variance": variance.tolist(),
        "excess_kurtosis": [float(v) if np.isfinite(v) else None for v in excess],
    }


def run(seed):
    data, metadata = generate(
        rows=60000, seed=seed, start_price=50000.0, transition_context_mode="episode_end"
    )
    x = np.column_stack([data[f] for f in FEATURES]).astype(np.float64)
    labels, episodes = data["gt_regime_id"], data["gt_episode_id"]
    if not np.isfinite(x).all():
        raise ValueError("generated feature path contains nonfinite values")
    split = 36000
    parameters = fit_controls(x[:split], labels[:split], episodes[:split])
    summaries = {}
    for name, a, b in (("train", 0, 36000), ("validation", 36000, 48000), ("test", 48000, 60000)):
        variants = residual_variants(x[a:b], labels[a:b], episodes[a:b], parameters)
        summaries[name] = {
            arm: [summarize(residual, labels[a:b], episodes[a:b], k, FEATURES) for k in range(4)]
            for arm, residual in variants.items()
        }
    serializable = {
        "mean": parameters["mean"].tolist(),
        "scale": parameters["scale"].tolist(),
        "state_mean": parameters["state_mean"].tolist(),
        "scalar": [{k: v.tolist() for k, v in row.items()} for row in parameters["scalar"]],
        "var1_coefficients": [v.tolist() for v in parameters["matrix"]],
    }
    print(f"seed={seed} residual diagnostics complete", flush=True)
    return {
        "generator_seed": seed,
        "dataset_schema": metadata["schema"],
        "feature_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "episodes_sha256": hashlib.sha256(episodes.tobytes()).hexdigest(),
        "parameters": serializable,
        "splits": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    payload = {
        "schema": "nhsmm-trading-residual-diagnostic-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "workers": 1,
            "source_sha256": {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [
                    __file__,
                    "scripts/generate_trading_regime_dataset.py",
                    "scripts/validate_trading_regime_usefulness.py",
                ]
            },
        },
        "features": list(FEATURES),
        "contract": {
            "rows": 60000,
            "generator_seeds": [424242, 424243, 424244],
            "start_price": 50000,
            "transition_context_mode": "episode_end",
            "split_ranges": {
                "train": [0, 36000],
                "validation": [36000, 48000],
                "test": [48000, 60000],
            },
            "training": "all chronological train rows; train-only standardization and oracle state means/AR/VAR fits; ridge 1e-4 fixed",
            "boundaries": "lags and predictors stay inside split and ground-truth episode; AR/VAR first episode rows omitted",
            "limits": "maintained synthetic v2 feature path, not real-market data or frozen legacy v1 NPZ; oracle-label diagnostics, not deployable predictors or calibrated likelihoods; no parameter/feature selection on validation/test",
        },
        "runs": [run(seed) for seed in (424242, 424243, 424244)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
