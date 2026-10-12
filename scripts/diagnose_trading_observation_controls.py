"""Separate observation overlap and temporal dependence using synthetic controls.

Oracle labels define observation-generating laws only, never model fitting.
These controls are not a new unlabeled trading-recovery acceptance result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_trading_decoder_duration import episode_median, score
from nhsmm.filtering import filter_model_sequence
from scripts.validate_trading_regime_usefulness import (
    FEATURES,
    K,
    RunBudget,
    _fit,
    _ll,
    _windows,
    _standardize,
    _standardize_context,
    _aligned_transition_mae,
)

ARMS = ("trading", "matched_iid", "matched_ar1", "separated_iid")
OBSERVATION_SEED = 56001
RHO = 0.9


def training_moments(x, labels):
    """Supervised diagnostic generator moments estimated only on training rows."""
    if x.shape[:-1] != labels.shape or not np.isfinite(x).all():
        raise ValueError("finite observations and matching labels required")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= K)):
        raise ValueError("integer state labels in range required")
    means, variances = [], []
    for state in range(K):
        subset = x[labels == state].astype(np.float64)
        if len(subset) < 2:
            raise ValueError("at least two training observations per state required")
        means.append(subset.mean(0))
        variances.append(np.maximum(subset.var(0), 1e-8))
    return np.stack(means), np.stack(variances)


def gaussian_control(labels, means, variances, *, seed=OBSERVATION_SEED, rho=0.0):
    """Stationary Gaussian residuals with fixed diagonal marginals and AR(1) rho."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("nonempty integer label vector required")
    if means.shape != variances.shape or means.ndim != 2 or means.shape[0] != K:
        raise ValueError("K-state mean/variance matrices required")
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("finite means and strictly positive variances required")
    if np.any((labels < 0) | (labels >= K)) or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("state range and abs(rho)<1 required")
    noise = np.random.default_rng(seed).normal(size=(len(labels), means.shape[1]))
    if rho:
        innovation_scale = np.sqrt(1 - rho * rho)
        for t in range(1, len(noise)):
            noise[t] = rho * noise[t - 1] + innovation_scale * noise[t]
    return (means[labels] + np.sqrt(variances[labels]) * noise).astype(np.float32)


def control_observations(data, train):
    """Keep complete latent/context trajectories and replace only observation columns."""
    means, variances = training_moments(train[0], train[2])
    strong_means = np.zeros((K, len(FEATURES)))
    strong_means[:, :K] = 4.0 * np.eye(K)
    strong_variances = np.full_like(strong_means, 0.35**2)
    labels = data["gt_regime_id"]
    controls = {
        "matched_iid": gaussian_control(labels, means, variances),
        "matched_ar1": gaussian_control(labels, means, variances, rho=RHO),
        "separated_iid": gaussian_control(labels, strong_means, strong_variances),
    }
    variants = {"trading": data}
    for name, observations in controls.items():
        variants[name] = {
            **data,
            **{feature: observations[:, i] for i, feature in enumerate(FEATURES)},
        }
    return variants, {
        "matched_means": means.tolist(),
        "matched_variances": variances.tolist(),
        "separated_means": strong_means.tolist(),
        "separated_variances": strong_variances.tolist(),
        "noise_seed": OBSERVATION_SEED,
        "ar1_rho": RHO,
    }


def serializable_score(truth, predicted):
    """Keep missing-state penalties while encoding absent run medians as null."""
    result = score(truth, predicted)
    result["run_medians"] = {
        key: value if np.isfinite(value) else None for key, value in result["run_medians"].items()
    }
    return result


def run(job):
    seed, path, iterations, arms = job
    torch.set_num_threads(1)
    with np.load(path) as z:
        data = {name: z[name] for name in z.files}
    n = len(data["close"])
    budget = RunBudget(240, 8, 4, iterations, 2)
    original_train = _windows(data, 0, int(0.6 * n), 8, window=240)
    variants, generator = control_observations(data, original_train)
    rows = {}
    for name in arms:
        variant = variants[name]
        train = _windows(variant, 0, int(0.6 * n), 8, window=240)
        test = _windows(variant, int(0.8 * n), n, 4, window=240)
        train_x, test_x = _standardize(train[0], test[0])
        train_c, test_c = _standardize_context(train[1], test[1])
        model = _fit(
            seed,
            train_x,
            train_c,
            context=True,
            hmm=False,
            emission_init_mode="kmeans_cluster_variance",
            budget=budget,
        )
        with torch.inference_mode():
            x, c = torch.tensor(test_x), torch.tensor(test_c)
            trace = filter_model_sequence(model, x, context=c)
            filtered = trace.state_posterior.argmax(-1).numpy()
            viterbi = np.stack([p.numpy() for p in model.predict(x, context=c, verbose=False)])
            duration = (
                model.dist.duration.log_matrix(
                    context=torch.zeros(1, 1, 1), T=1, soft_dmax=model.duration_logits_bias
                )
                .exp()
                .reshape(K, model.config.max_duration)
                .numpy()
            )
            tail = model.dist.duration.tail_probability().numpy()
        filtered_score = serializable_score(test[2], filtered)
        rows[name] = {
            "filter": filtered_score,
            "viterbi": serializable_score(test[2], viterbi),
            "train_ll_per_row": _ll(model, train_x, train_c),
            "test_ll_per_row": _ll(model, test_x, test_c),
            "transition_mae": _aligned_transition_mae(
                model, test_c, test[2], test[3], test[5], filtered_score["permutation"]
            ),
            "learned_episode_medians_zero_context": [
                episode_median(duration[state], float(tail[state])) for state in range(K)
            ],
            "true_switches": int(np.count_nonzero(test[2][:, 1:] != test[2][:, :-1])),
            "observation_sha256": hashlib.sha256(
                np.column_stack([variant[f] for f in FEATURES]).astype(np.float32).tobytes()
            ).hexdigest(),
        }
        print(f"seed={seed} arm={name} complete", flush=True)
    return {"seed": seed, "arms": rows, "generator": generator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    try:
        seeds = tuple(int(v) for v in args.seeds.split(","))
        arms = tuple(args.arms.split(","))
        if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1 or args.max_iter < 1:
            raise ValueError("distinct seeds and positive workers/iterations required")
        if not arms or len(set(arms)) != len(arms) or not set(arms).issubset(ARMS):
            raise ValueError("distinct known arms required")
    except ValueError as exc:
        parser.error(str(exc))
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end dataset required")
    jobs = [(seed, args.dataset, args.max_iter, arms) for seed in seeds]
    if args.workers == 1:
        rows = [run(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds))
        ) as pool:
            rows = list(pool.map(run, jobs))
    metrics = ("accuracy", "ari", "boundary_f1_tol5", "run_duration_rel_error", "switches")
    summary = {
        name: {
            decoder: {
                metric: float(np.median([r["arms"][name][decoder][metric] for r in rows]))
                for metric in metrics
            }
            for decoder in ("filter", "viterbi")
        }
        for name in arms
    }
    payload = {
        "schema": "nhsmm-trading-observation-controls-pilot-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "max_iter": args.max_iter,
        "window": 240,
        "train_windows": 8,
        "eval_windows": 4,
        "transition_refine_steps": 2,
        "full_protocol": False,
        "runs": rows,
        "summary": summary,
        "contract": {
            "labels": "oracle labels define synthetic generating laws; no labels passed to optimize or used for rule selection",
            "matched_moments": "supervised diagnostic estimates from the eight training windows only",
            "generative_controls": "same frozen latent/context trajectories and random Gaussian innovations for all optimization seeds",
            "matched_iid_vs_ar1": "same population per-state diagonal Gaussian marginals, rho=0 vs rho=0.9",
            "trading_vs_matched_iid": "changes non-Gaussianity, within-row dependence and serial dependence simultaneously; not an isolated test of overlap",
            "separated_iid": "fixed one-hot means of amplitude 4 in first four features, independent noise std 0.35",
            "temporal_law": "original non-geometric durations and zero-diagonal context transitions retained; not an exactly model-conform duration-tail test",
            "duration": "hard run recovery differs from recovery of learned duration parameters",
            "likelihood": "only within-arm train/OOS density measurements; different observations preclude cross-arm improvement claims",
            "status": "exploratory diagnostic controls, not trading recovery acceptance or new defaults",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
