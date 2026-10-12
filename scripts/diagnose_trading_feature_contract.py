"""Fixed feature/emission pilot with common held-out state and return targets."""

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
from scripts.diagnose_model_conform_duration import FixedTemperatureNHSMM, optimization_evidence
from scripts.diagnose_trading_feature_redundancy import REDUCED, fit_whitening
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

ARMS = {
    "full8_gaussian": (FEATURES, "gaussian", False),
    "reduced4_gaussian": (REDUCED, "gaussian", False),
    "return_volume2_gaussian": (("log_return_1", "volume_z_48"), "gaussian", False),
    "whiten8_gaussian": (FEATURES, "gaussian", True),
    "full8_studentt": (FEATURES, "studentt", False),
    "reduced4_studentt": (REDUCED, "studentt", False),
}
WARMUP = 64


class MatchedFamilyModel(FixedTemperatureNHSMM):
    """Script-only Student-t base variance matching at each restart initialization."""

    @torch.no_grad()
    def _initialize_emission_from_observations(self, observations, *, cluster_variance=False):
        super()._initialize_emission_from_observations(
            observations, cluster_variance=cluster_variance
        )
        emission = self.dist.emission
        if emission.emission_type != "studentt":
            return
        x = observations.to(emission.loc)
        labels = torch.cdist(x, emission.loc).argmin(-1)
        global_variance = x.var(0, unbiased=False)
        variance = torch.stack(
            [
                (
                    x[labels == k].var(0, unbiased=False)
                    if int((labels == k).sum()) > 1
                    else global_variance
                )
                for k in range(self.config.n_states)
            ]
        ).clamp(0.05, 5.0)
        df = torch.nn.functional.softplus(emission.dof) + 2
        scale = torch.sqrt(variance * (df[:, None] - 2) / df[:, None])
        emission.scale_param.copy_(scale + torch.log(-torch.expm1(-scale)))


def aligned_probability(posterior, permutation):
    return posterior[..., np.argsort(permutation)]


def state_scores(posterior, labels, permutation):
    probability = aligned_probability(
        np.asarray(posterior[:, WARMUP:], dtype=np.float64), permutation
    )
    truth = labels[:, WARMUP:]
    chosen = np.take_along_axis(probability, truth[:, :, None], axis=-1).squeeze(-1)
    return {
        "accuracy": float(np.mean(probability.argmax(-1) == truth)),
        "brier": float(((probability - np.eye(4)[truth]) ** 2).sum(-1).mean()),
        "state_log_score": float(np.log(np.maximum(chosen, 1e-300)).mean()),
        "zero_true_state_probabilities": int(np.sum(chosen == 0)),
        "rows": int(truth.size),
    }


def return_design(returns, posterior, mean, scale):
    previous = (returns[:, WARMUP:-1] - mean) / scale
    pieces = [np.ones((*previous.shape, 1)), previous[:, :, None]]
    if posterior is not None:
        pieces.append(posterior[:, WARMUP:-1])
    return np.concatenate(pieces, axis=-1).reshape(-1, 2 if posterior is None else 6)


def fit_return_head(returns, posterior):
    """Supervised downstream calibration only; gradients never reach the HSMM."""
    target = returns[:, WARMUP + 1 :].reshape(-1).astype(float)
    mean, scale = float(target.mean()), max(float(target.std()), 1e-8)
    design = return_design(returns, posterior, mean, scale)
    y = (target - mean) / scale
    penalty = np.diag([0.0] + [1e-3] * (design.shape[1] - 1))
    beta = np.linalg.solve(design.T @ design / len(y) + penalty, design.T @ y / len(y))
    residual = y - design @ beta
    pooled = max(float(np.mean(residual**2)), 1e-6)
    if posterior is None:
        variances = np.array([pooled])
    else:
        weights = posterior[:, WARMUP:-1].reshape(-1, 4)
        counts = weights.sum(0)
        variances = np.divide(
            weights.T @ residual**2, counts, out=np.full(4, pooled), where=counts > 1e-8
        )
        variances = np.maximum(variances, 1e-6)
    return {"mean": mean, "scale": scale, "beta": beta, "variances": variances}


def return_scores(returns, posterior, parameters):
    design = return_design(returns, posterior, parameters["mean"], parameters["scale"])
    target = returns[:, WARMUP + 1 :].reshape(-1).astype(float)
    prediction = design @ parameters["beta"]
    if posterior is None:
        variance = np.full(len(target), parameters["variances"][0])
    else:
        weights = posterior[:, WARMUP:-1].reshape(-1, 4)
        variance = weights @ parameters["variances"]
    normalized = (target - parameters["mean"]) / parameters["scale"]
    score = -0.5 * (
        np.log(2 * math.pi * variance) + (normalized - prediction) ** 2 / variance
    ) - math.log(parameters["scale"])
    error = target - (parameters["mean"] + parameters["scale"] * prediction)
    return {
        "next_return_log_score": float(score.mean()),
        "next_return_rmse": float(np.sqrt(np.mean(error**2))),
        "pairs": len(target),
    }


def run(seed, base):
    torch.set_num_threads(1)
    data, metadata = generate(
        rows=60000, seed=424242, start_price=50000.0, transition_context_mode="episode_end"
    )
    raw = np.column_stack([data[f] for f in FEATURES]).astype(float)
    prior_path = Path("docs/validation/results/trading-residual-step6-native-20261012.json")
    prior_bytes = prior_path.read_bytes()
    prior = next(r for r in json.loads(prior_bytes)["runs"] if r["generator_seed"] == 424242)
    if hashlib.sha256(raw.tobytes()).hexdigest() != prior["feature_sha256"]:
        raise AssertionError("frozen Step 6 dataset mismatch")
    train = _windows(data, 0, 36000, 8, window=240)
    test = _windows(data, 48000, 60000, 4, window=240)
    tx, vx = _standardize(train[0], test[0])
    tc, vc = _standardize_context(train[1], test[1])
    whitening = fit_whitening(tx)
    rows = {}
    budget = RunBudget(
        window=240, train_windows=8, eval_windows=4, max_iter=40, transition_refine_steps=0
    )
    baseline = fit_return_head(train[0][:, :, 0], None)
    for name, (features, family, whiten) in ARMS.items():
        indices = [FEATURES.index(f) for f in features]
        x = whitening.apply(tx) if whiten else tx[:, :, indices]
        v = whitening.apply(vx) if whiten else vx[:, :, indices]
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        cfg = _config(
            seed,
            context=True,
            hmm=False,
            emission_init_mode="kmeans" if family == "studentt" else "kmeans_cluster_variance",
            budget=budget,
        )
        cfg = cfg.with_overrides(n_features=len(features), emission_type=family)
        model = MatchedFamilyModel(cfg, device="cpu")
        model.initialize_distributions()
        model.optimize(torch.tensor(x), context=torch.tensor(tc))
        model.eval()
        evidence = optimization_evidence(model, 40)
        with torch.inference_mode():
            train_p = filter_model_sequence(
                model, torch.tensor(x), context=torch.tensor(tc)
            ).state_posterior.numpy()
            test_p = filter_model_sequence(
                model, torch.tensor(v), context=torch.tensor(vc)
            ).state_posterior.numpy()
        permutation, train_accuracy = _best_perm(
            train[2][:, WARMUP:], train_p[:, WARMUP:].argmax(-1)
        )
        head = fit_return_head(train[0][:, :, 0], train_p)
        rows[name] = {
            "features": list(features),
            "family": family,
            "whitening": whiten,
            "train_mapping_accuracy": train_accuracy,
            "train_permutation": list(permutation),
            "state": state_scores(test_p, test[2], permutation),
            "return": return_scores(test[0][:, :, 0], test_p, head),
            "optimization": evidence,
            "return_head": {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in head.items()
            },
        }
        print(f"seed={seed} arm={name} complete", flush=True)
    return {
        "fit_seed": seed,
        "base_commit": base,
        "dataset_schema": metadata["schema"],
        "dataset_feature_sha256": prior["feature_sha256"],
        "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "no_regime_return_baseline": return_scores(test[0][:, :, 0], None, baseline),
        "arms": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    payload = {
        "schema": "nhsmm-trading-feature-contract-pilot-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [
                    __file__,
                    "scripts/generate_trading_regime_dataset.py",
                    "scripts/validate_trading_regime_usefulness.py",
                    "scripts/diagnose_trading_feature_redundancy.py",
                    "scripts/diagnose_model_conform_duration.py",
                ]
            },
        },
        "contract": {
            "dataset_seed": 424242,
            "rows": 60000,
            "train_range": [0, 36000],
            "test_range": [48000, 60000],
            "windows": "8 train/4 test windows of 240 rows; deterministic benchmark anchors",
            "model": "K4 D64 plus tail, external macro context; 40 actual updates, one restart, temperature1; no post-transition refinement",
            "initialization": "KMeans cluster variance; Student-t base scale adjusted to same cluster marginal variance; script-only hook",
            "labels": "training labels freeze state permutation for evaluation only; no labels in HSMM or return-head parameter inputs",
            "common_targets": "state probability scores after 64-row warmup; raw next log_return_1 Gaussian score/RMSE, 175 causal pairs/window",
            "return_head": "train-only downstream ridge1e-3 mean fit on current return plus frozen current posterior; posterior-weighted training residual variance; no gradients into representation",
            "comparison": "input-space densities across feature dimensions are excluded; no OOS selection/default change",
            "limits": "bounded full-model pilot on one synthetic dataset, three fit seeds; family initialization controlled but hypothesis classes differ; no trading-utility/full acceptance claim",
        },
        "run": run(args.seed, args.base_commit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                a: {"state": v["state"], "return": v["return"]}
                for a, v in payload["run"]["arms"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
