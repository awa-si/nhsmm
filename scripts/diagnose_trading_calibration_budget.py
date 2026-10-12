"""Fixed 40/120-update trajectories and train-only nested head calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import filter_model_sequence
from scripts.diagnose_model_conform_duration import optimization_evidence
from scripts.diagnose_trading_feature_contract import (
    MatchedFamilyModel,
    fit_return_head,
    return_scores,
)
from scripts.diagnose_trading_information_baselines import (
    FIT_SEED,
    WARMUP,
    calibrated_result,
    observable_design,
    raw_state_scores,
    score_head,
    serial_head,
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

RIDGES = (0.001, 0.01, 0.1)
UPDATES = 120


class BudgetCheckpointModel(MatchedFamilyModel):
    """Capture weights before update 41, after exactly 40 completed updates."""

    def __init__(self, *args, checkpoint_iteration=40, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_iteration = checkpoint_iteration
        self.checkpoint = None
        self.logged_iterations = set()

    def _compute_loss(self, *args, **kwargs):
        iteration = kwargs.get("it", 0)
        if iteration == self.checkpoint_iteration and self.checkpoint is None:
            self.checkpoint = {k: v.detach().clone() for k, v in self.state_dict().items()}
        if iteration in (20, 40, 80) and iteration not in self.logged_iterations:
            self.logged_iterations.add(iteration)
            print(f"completed updates={iteration}", flush=True)
        return super()._compute_loss(*args, **kwargs)


def window_folds(n_windows):
    if n_windows < 2:
        raise ValueError("at least two windows required for cross-fitting")
    return np.array_split(np.arange(n_windows), min(4, n_windows))


def ridge_fit(design, target, ridge):
    penalty = np.diag([0.0] + [ridge] * (design.shape[-1] - 1))
    x, y = design.reshape(-1, design.shape[-1]), target.reshape(-1)
    return np.linalg.solve(x.T @ x / len(y) + penalty, x.T @ y / len(y))


def oof_mean_raw(design, target, ridge):
    """Each window's raw prediction excludes all targets in its held-out block."""
    predictions = np.empty_like(target, dtype=np.float64)
    for held in window_folds(len(target)):
        training = np.setdiff1d(np.arange(len(target)), held)
        y = target[training]
        mean, scale = float(y.mean()), max(float(y.std()), 1e-8)
        beta = ridge_fit(design[training], (y - mean) / scale, ridge)
        predictions[held] = mean + scale * (design[held] @ beta)
    return predictions


def fit_crossfit_head(design, target, ridge):
    """Fit mean on all training targets and variance on out-of-fold mean errors."""
    mean, scale = float(target.mean()), max(float(target.std()), 1e-8)
    beta = ridge_fit(design, (target - mean) / scale, ridge)
    squared = ((target - oof_mean_raw(design, target, ridge)) / scale) ** 2
    gamma = ridge_fit(design, np.log(np.maximum(squared, 1e-6)), ridge)
    preliminary = np.exp(np.clip(design @ gamma, -12, 12))
    smear = max(float(np.mean(squared / preliminary)), 1e-6)
    return {"mean": mean, "scale": scale, "beta": beta, "gamma": gamma, "smear": smear}


def target_returns(target):
    """Adapter for the maintained raw-unit score on an already paired target."""
    values = np.zeros((len(target), WARMUP + 1 + target.shape[1]), dtype=np.float64)
    values[:, WARMUP + 1 :] = target
    return values


def tune_head(design, target):
    """Nested window CV; candidate residual cross-fitting sees outer-training only."""
    candidate_scores = []
    for ridge in RIDGES:
        folds = []
        for held in window_folds(len(target)):
            training = np.setdiff1d(np.arange(len(target)), held)
            head = fit_crossfit_head(design[training], target[training], ridge)
            folds.append(
                score_head(
                    design[held].reshape(-1, design.shape[-1]), target_returns(target[held]), head
                )["next_return_log_score"]
            )
        candidate_scores.append(
            {"ridge": ridge, "fold_log_scores": folds, "mean_log_score": float(np.mean(folds))}
        )
    # Stable predeclared ordering breaks exact ties; test data are not accepted.
    selected = max(candidate_scores, key=lambda row: row["mean_log_score"])["ridge"]
    return fit_crossfit_head(design, target, selected), {
        "selected_ridge": selected,
        "candidate_scores": candidate_scores,
        "outer_held_windows": [held.tolist() for held in window_folds(len(target))],
    }


def tuned_result(train_design, test_design, train_returns, test_returns):
    target = train_returns[:, WARMUP + 1 :].astype(np.float64)
    shaped = train_design.reshape(len(target), target.shape[1], -1)
    head, selection = tune_head(shaped, target)
    return {
        "return": score_head(test_design, test_returns, head),
        "head": serial_head(head),
        "selection": selection,
    }


def run(generator_seed):
    torch.set_num_threads(1)
    data, metadata = generate(
        rows=60000, seed=generator_seed, start_price=50000.0, transition_context_mode="episode_end"
    )
    prior_path = Path(
        f"docs/validation/results/information-baselines-step8-data{generator_seed}-native-20261012.json"
    )
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)["run"]
    raw = np.column_stack([data[f] for f in FEATURES]).astype(float)
    if hashlib.sha256(raw.tobytes()).hexdigest() != prior["dataset_feature_sha256"]:
        raise AssertionError("frozen Step 8 dataset mismatch")
    train = _windows(data, 0, 36000, 8, window=240)
    test = _windows(data, 48000, 60000, 4, window=240)
    tx, vx = _standardize(train[0], test[0])
    tc, vc = _standardize_context(train[1], test[1])
    tr, vr = train[0][:, :, 0], test[0][:, :, 0]
    baselines = {}
    for name, lags in (("instant", (0,)), ("lagged", (0, 1, 3, 12, 48))):
        baselines[name] = tuned_result(
            observable_design(tx, tc, lags=lags), observable_design(vx, vc, lags=lags), tr, vr
        )
    budget = RunBudget(
        window=240, train_windows=8, eval_windows=4, max_iter=UPDATES, transition_refine_steps=0
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
        model = BudgetCheckpointModel(cfg, device="cpu")
        model.initialize_distributions()
        model.optimize(torch.tensor(tx), context=torch.tensor(tc))
        model.eval()
        evidence = optimization_evidence(model, UPDATES)
        if model.checkpoint is None:
            raise AssertionError("40-update checkpoint missing")
        final = {k: v.detach().clone() for k, v in model.state_dict().items()}
        trajectory = model._convergence.scores[:, 0]
        trajectory = trajectory[torch.isfinite(trajectory)].tolist()
        snapshots = {}
        for updates, weights in ((40, model.checkpoint), (UPDATES, final)):
            model.load_state_dict(weights)
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
            train_design, test_design = observable_design(tx, tc, train_p), observable_design(
                vx, vc, test_p
            )
            fixed = calibrated_result(train_design, test_design, tr, vr, hetero=True)
            tuned = tuned_result(train_design, test_design, tr, vr)
            state = raw_state_scores(test_log, test[2], permutation)
            if updates == 40:
                for key in ("accuracy", "brier", "state_log_score"):
                    if not np.isclose(
                        state[key], prior["arms"][family]["state"][key], rtol=0, atol=1e-6
                    ):
                        raise AssertionError(f"Step 8 checkpoint mismatch: {family} {key}")
                for key in ("next_return_log_score", "next_return_rmse"):
                    if not np.isclose(
                        fixed["return"][key],
                        prior["arms"][family]["lagged_plus_posterior_heteroscedastic"]["return"][
                            key
                        ],
                        rtol=0,
                        atol=1e-7,
                    ):
                        raise AssertionError(f"Step 8 forecast checkpoint mismatch: {family} {key}")
            snapshots[str(updates)] = {
                "state": state,
                "train_permutation": list(permutation),
                "train_mapping_accuracy": train_accuracy,
                "fixed_head": fixed,
                "tuned_crossfit_head": tuned,
                "step7_return": return_scores(vr, test_p, fit_return_head(tr, train_p)),
                "increment_vs_tuned_lagged": {
                    k: tuned["return"][k] - baselines["lagged"]["return"][k]
                    for k in ("next_return_log_score", "next_return_rmse")
                },
            }
        arms[family] = {
            "optimization": evidence,
            "pre_update_train_log_likelihood": trajectory,
            "final_train_log_likelihood": model._restart_scores[0],
            "snapshots": snapshots,
        }
        print(f"dataset={generator_seed} family={family} complete", flush=True)
    return {
        "generator_seed": generator_seed,
        "fit_seed": FIT_SEED,
        "dataset_schema": metadata["schema"],
        "dataset_feature_sha256": prior["dataset_feature_sha256"],
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
        "scripts/diagnose_trading_calibration_budget.py",
        "scripts/diagnose_trading_information_baselines.py",
        "scripts/diagnose_trading_feature_contract.py",
        "scripts/diagnose_model_conform_duration.py",
        "scripts/generate_trading_regime_dataset.py",
        "scripts/validate_trading_regime_usefulness.py",
        "nhsmm/models/base.py",
    ]
    payload = {
        "schema": "nhsmm-trading-calibration-budget-v1",
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
            "windows": "same Step 8 train/test anchors, 8/4 windows240, warmup64, 1400 train/700 test pairs",
            "budgets": [40, UPDATES],
            "checkpoint": "weights before update41; same uninterrupted trajectory; checked against Step 8 scores",
            "model": "unchanged K4 D64+tail, full8, macro external context, temperature1, n_init1, no scheduler/early stop/transition refinement",
            "head_selection": "four contiguous window-block outer training folds; nested mean residual cross-fitting only in outer training; maximum mean raw-unit CV log score over fixed ridge grid",
            "ridge_grid": list(RIDGES),
            "variance": "ridge log squared OOF mean residuals; in-sample smearing for variance regression, logvar clip[-12,12], normalized variance floor1e-6",
            "representation": "HSMM and observable normalization fitted on all training windows; head CV is conditional on frozen representation, not end-to-end cross-fitting or walk-forward",
            "test": "no external test or validation segment in selection; train permutation only; all methods score same raw targets",
            "limits": "synthetic sparse-window pilot, one fit seed; baseline fixed lags48 vs full HSMM prefix; 120 updates does not imply convergence; nested head folds share unsupervised representation; no statistical significance or trading utility claim",
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
                    f: {
                        u: {"state": s["state"], "tuned": s["tuned_crossfit_head"]["return"]}
                        for u, s in a["snapshots"].items()
                    }
                    for f, a in payload["run"]["arms"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
