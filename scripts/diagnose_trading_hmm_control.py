"""Decision-focused age-free HMM control against frozen Step 9 HSMM evidence."""

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
from scripts.diagnose_trading_calibration_budget import (
    BudgetCheckpointModel,
    fit_crossfit_head,
)
from scripts.diagnose_trading_feature_contract import (
    MatchedFamilyModel,
    fit_return_head,
    return_scores,
)
from scripts.diagnose_trading_information_baselines import (
    FIT_SEED,
    WARMUP,
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

RIDGE = 0.1
UPDATES = 120


def hsmm_initial_states(config, observations):
    """Recreate the Step 9 restart initialization without optimizing a prototype."""
    np.random.seed(FIT_SEED)
    random.seed(FIT_SEED)
    torch.manual_seed(FIT_SEED)
    prototype = MatchedFamilyModel(config, device="cpu")
    prototype.initialize_distributions()
    prototype._initialize_run_state(
        0,
        observations=prototype._training_observations(observations),
        emission_init_mode=config.emission_init_mode,
    )
    return [
        {k: v.detach().clone() for k, v in distribution.state_dict().items()}
        for distribution in (prototype.dist.emission, prototype.dist.initial)
    ]


class HMMControlModel(BudgetCheckpointModel):
    """Match all emission and initial-prior restart parameters to the HSMM."""

    def __init__(self, *args, emission_state, initial_state, **kwargs):
        super().__init__(*args, **kwargs)
        self.matched_emission_state = emission_state
        self.matched_initial_state = initial_state

    def _initialize_emission_from_observations(self, observations, *, cluster_variance=False):
        self.dist.emission.load_state_dict(self.matched_emission_state)

    def _initialize_run_state(self, *args, **kwargs):
        super()._initialize_run_state(*args, **kwargs)
        self.dist.initial.load_state_dict(self.matched_initial_state)


def logsumexp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def numpy_hmm(log_initial, log_transition, emission):
    """Independent state-only recursion; row t-1 drives the boundary into row t."""
    emission = np.asarray(emission, dtype=np.float64)
    log_initial = np.asarray(log_initial, dtype=np.float64)
    log_transition = np.asarray(log_transition, dtype=np.float64)
    alpha = log_initial + emission[:, 0]
    evidence = logsumexp(alpha, -1)
    posterior = np.empty_like(emission)
    posterior[:, 0] = alpha - evidence[:, None]
    for t in range(1, emission.shape[1]):
        predicted = logsumexp(posterior[:, t - 1, :, None] + log_transition[:, t - 1], 1)
        alpha = predicted + emission[:, t]
        normalizer = logsumexp(alpha, -1)
        posterior[:, t] = alpha - normalizer[:, None]
        evidence += normalizer
    return posterior, evidence


def checked_hmm_filter(model, observations, contexts):
    """Check production D1 filtering against NumPy using the same learned scores."""
    if model.config.max_duration != 1 or model.config.duration_tail:
        raise ValueError("age-free control requires D1 with tail disabled")
    with torch.inference_mode():
        sequence = model._build_sequence_set(observations, context=contexts)
        t = sequence.log_probs.shape[1]
        log_initial = model.dist.initial.log_matrix(context=sequence.canonical, T=t)
        log_transition = model.dist.transition.log_matrix(context=sequence.contexts, T=t)
        log_duration = model.dist.duration.log_matrix(
            context=sequence.contexts, T=t, soft_dmax=model.duration_logits_bias
        )
        torch.testing.assert_close(log_duration, torch.zeros_like(log_duration), rtol=0, atol=0)
        expected, evidence = numpy_hmm(
            log_initial[:, 0].numpy(),
            log_transition[:, :, :, 0].numpy(),
            sequence.log_probs.numpy(),
        )
        actual = torch.logsumexp(
            filter_model_sequence(model, observations, context=contexts).log_posterior, -1
        ).numpy()
        error = float(np.max(np.abs(actual.astype(float) - expected)))
        likelihood_error = float(
            abs(
                float(model.log_likelihood(observations, context=contexts, reduce=True))
                - evidence.sum()
            )
            / observations.shape[0]
            / observations.shape[1]
        )
    if error > 1e-4 or likelihood_error > 2e-5:
        raise AssertionError(
            f"HMM recursion mismatch: posterior={error} likelihood/row={likelihood_error}"
        )
    return actual, {"max_log_posterior_error": error, "likelihood_error_per_row": likelihood_error}


def run(seed):
    torch.set_num_threads(1)
    prior_path = Path(
        f"docs/validation/results/calibration-budget-step9-data{seed}-native-20261012.json"
    )
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)["run"]
    # This penalty was selected by every Step 9 internal training fold pipeline.
    for row in prior["baselines"].values():
        if row["selection"]["selected_ridge"] != RIDGE:
            raise AssertionError("baseline training-selected penalty differs")
    for arm in prior["arms"].values():
        for snapshot in arm["snapshots"].values():
            if snapshot["tuned_crossfit_head"]["selection"]["selected_ridge"] != RIDGE:
                raise AssertionError("HSMM training-selected penalty differs")
    data, metadata = generate(
        rows=60000, seed=seed, start_price=50000.0, transition_context_mode="episode_end"
    )
    raw = np.column_stack([data[f] for f in FEATURES]).astype(float)
    if hashlib.sha256(raw.tobytes()).hexdigest() != prior["dataset_feature_sha256"]:
        raise AssertionError("frozen dataset mismatch")
    train = _windows(data, 0, 36000, 8, window=240)
    test = _windows(data, 48000, 60000, 4, window=240)
    tx, vx = _standardize(train[0], test[0])
    tc, vc = _standardize_context(train[1], test[1])
    tr, vr = train[0][:, :, 0], test[0][:, :, 0]
    budget = RunBudget(
        window=240, train_windows=8, eval_windows=4, max_iter=UPDATES, transition_refine_steps=0
    )
    arms = {}
    for family in ("gaussian", "studentt"):
        cfg = _config(
            FIT_SEED,
            context=True,
            hmm=False,
            emission_init_mode="kmeans" if family == "studentt" else "kmeans_cluster_variance",
            budget=budget,
        ).with_overrides(n_features=len(FEATURES), emission_type=family)
        emission_state, initial_state = hsmm_initial_states(cfg, torch.tensor(tx))
        model = HMMControlModel(
            cfg.with_overrides(max_duration=1, duration_tail=False),
            device="cpu",
            emission_state=emission_state,
            initial_state=initial_state,
        )
        model.initialize_distributions()
        model.optimize(torch.tensor(tx), context=torch.tensor(tc))
        model.eval()
        evidence = optimization_evidence(model, UPDATES)
        if model.checkpoint is None:
            raise AssertionError("missing 40-update checkpoint")
        final = {k: v.detach().clone() for k, v in model.state_dict().items()}
        snapshots = {}
        for updates, weights in ((40, model.checkpoint), (UPDATES, final)):
            model.load_state_dict(weights)
            train_log, train_check = checked_hmm_filter(model, torch.tensor(tx), torch.tensor(tc))
            test_log, test_check = checked_hmm_filter(model, torch.tensor(vx), torch.tensor(vc))
            train_p, test_p = np.exp(train_log), np.exp(test_log)
            permutation, train_accuracy = _best_perm(
                train[2][:, WARMUP:], train_log[:, WARMUP:].argmax(-1)
            )
            design = observable_design(tx, tc, train_p)
            target = tr[:, WARMUP + 1 :].astype(np.float64)
            head = fit_crossfit_head(
                design.reshape(len(target), target.shape[1], -1), target, RIDGE
            )
            score = score_head(observable_design(vx, vc, test_p), vr, head)
            state = raw_state_scores(test_log, test[2], permutation)
            hsmm = prior["arms"][family]["snapshots"][str(updates)]
            snapshots[str(updates)] = {
                "state": state,
                "train_permutation": list(permutation),
                "train_mapping_accuracy": train_accuracy,
                "return": score,
                "head": serial_head(head),
                "step7_return": return_scores(vr, test_p, fit_return_head(tr, train_p)),
                "numpy_check": {"train": train_check, "test": test_check},
                "hmm_minus_hsmm": {
                    "state_accuracy": state["accuracy"] - hsmm["state"]["accuracy"],
                    "state_log_score": state["state_log_score"] - hsmm["state"]["state_log_score"],
                    "next_return_log_score": score["next_return_log_score"]
                    - hsmm["tuned_crossfit_head"]["return"]["next_return_log_score"],
                    "next_return_rmse": score["next_return_rmse"]
                    - hsmm["tuned_crossfit_head"]["return"]["next_return_rmse"],
                },
            }
        arms[family] = {"optimization": evidence, "snapshots": snapshots}
        print(f"dataset={seed} HMM family={family} complete", flush=True)
    return {
        "generator_seed": seed,
        "fit_seed": FIT_SEED,
        "dataset_schema": metadata["schema"],
        "dataset_feature_sha256": prior["dataset_feature_sha256"],
        "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "hsmm_reference": {
            f: {
                u: {
                    "state": s["state"],
                    "return": s["tuned_crossfit_head"]["return"],
                    "step7_return": s["step7_return"],
                }
                for u, s in a["snapshots"].items()
            }
            for f, a in prior["arms"].items()
        },
        "direct_reference": {k: v["return"] for k, v in prior["baselines"].items()},
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
        "scripts/diagnose_trading_hmm_control.py",
        "scripts/diagnose_trading_calibration_budget.py",
        "scripts/diagnose_trading_feature_contract.py",
        "scripts/diagnose_trading_information_baselines.py",
        "scripts/diagnose_model_conform_duration.py",
        "scripts/generate_trading_regime_dataset.py",
        "scripts/validate_trading_regime_usefulness.py",
        "nhsmm/models/base.py",
        "nhsmm/filtering.py",
    ]
    payload = {
        "schema": "nhsmm-trading-hmm-control-v1",
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
            "datasets": [424242, 424243, 424244],
            "fit_seed": FIT_SEED,
            "budgets": [40, UPDATES],
            "windows": "same frozen Step 9 8/4 windows240, warmup64; 704 state rows, 700 future-return targets",
            "hmm": "production D1, tail disabled; arbitrary self-transitions retained; no age-dependent transition rows or explicit duration memory",
            "matching": "copy full initial emission and initial-prior restart parameters from reconstructed Step 9 HSMM initializer; same features, macro, optimizer/lr/temperature/budget",
            "head": "fixed ridge0.1 from Step 9 internal training selection, unchanged cross-fitted mean-residual variance calibration; no new hyperparameter search",
            "independence_check": "NumPy state-only HMM recurrence with learned scores; posterior and public likelihood checked for train/test at each budget",
            "limits": "single fit seed, synthetic sparse windows, different transition capacity and random networks; matched emission/prior initialization is not full parameter identity; shared training representation, no walk-forward/PnL/convergence claim; compare common targets rather than duration parameter values",
        },
        "run": run(args.generator_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                f: {
                    u: {
                        "state": s["state"],
                        "return": s["return"],
                        "hmm_minus_hsmm": s["hmm_minus_hsmm"],
                    }
                    for u, s in a["snapshots"].items()
                }
                for f, a in payload["run"]["arms"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
