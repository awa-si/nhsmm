"""Variance-encoding regression and predeclared inference-only emission-weight probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from contextlib import contextmanager
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

WEIGHTS = (1.0, 0.5, 0.25, 0.125)


@contextmanager
def emission_weight(model, weight):
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be finite and positive")
    original = model._build_sequence_set

    def weighted(*args, **kwargs):
        sequence = original(*args, **kwargs)
        sequence.log_probs = sequence.log_probs * weight
        return sequence

    model._build_sequence_set = weighted
    try:
        yield
    finally:
        model._build_sequence_set = original


class LegacyVarianceNHSMM(NHSMM):
    """Reproduce the previous wrong log encoding, confined to this diagnostic."""

    @torch.no_grad()
    def _initialize_emission_from_observations(self, observations, *, cluster_variance=False):
        super()._initialize_emission_from_observations(
            observations, cluster_variance=cluster_variance
        )
        if cluster_variance:
            centers = self.dist.emission.mu
            labels = torch.cdist(observations, centers).argmin(-1)
            global_var = observations.var(0, unbiased=False)
            variance = torch.stack(
                [
                    (
                        observations[labels == k].var(0, unbiased=False)
                        if int((labels == k).sum()) > 1
                        else global_var
                    )
                    for k in range(self.config.n_states)
                ]
            ).clamp(0.05, 5.0)
            self.dist.emission.log_var.copy_(variance.log())


def fit(seed, x, c, budget, legacy):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model_type = LegacyVarianceNHSMM if legacy else NHSMM
    model = model_type(
        _config(
            seed,
            context=True,
            hmm=False,
            emission_init_mode="kmeans_cluster_variance",
            budget=budget,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.optimize(torch.tensor(x), context=torch.tensor(c))
    model.eval()
    return model


def feature_structure(train):
    """Training-only correlations; pooled regime mixtures can also cause dependence."""
    flat = train.reshape(-1, train.shape[-1])
    correlation = np.corrcoef(flat, rowvar=False)
    lag1 = [
        float(np.corrcoef(train[:, :-1, i].ravel(), train[:, 1:, i].ravel())[0, 1])
        for i in range(train.shape[-1])
    ]
    return {
        "features": list(FEATURES),
        "correlation": correlation.tolist(),
        "within_window_lag1_correlation": lag1,
    }


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
    x, c = torch.tensor(test_x), torch.tensor(test_c)
    arms = []
    for legacy in (True, False):
        model = fit(seed, train_x, train_c, budget, legacy)
        ll = _ll(model, test_x, test_c)
        for weight in ((1.0,) if legacy else WEIGHTS):
            with emission_weight(model, weight), torch.inference_mode():
                posterior = filter_model_sequence(model, x, context=c).state_posterior.numpy()
            pred = posterior.argmax(-1)
            row = score(test[2], pred)
            row.update(
                encoding="legacy_log" if legacy else "inverse_softplus",
                weight=weight,
                posterior_entropy=float(
                    np.mean(-np.sum(posterior * np.log(np.maximum(posterior, 1e-30)), axis=-1))
                ),
                proper_unweighted_test_ll_per_row=ll if weight == 1 else None,
            )
            arms.append(row)
    return {
        "seed": seed,
        "arms": arms,
        "training_feature_structure": feature_structure(train_x),
        "true_switches": int(np.count_nonzero(test[2][:, 1:] != test[2][:, :-1])),
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
    for encoding, weights in (("legacy_log", (1.0,)), ("inverse_softplus", WEIGHTS)):
        for weight in weights:
            group = [
                a
                for r in rows
                for a in r["arms"]
                if a["encoding"] == encoding and a["weight"] == weight
            ]
            metrics = (
                "accuracy",
                "ari",
                "boundary_f1_tol5",
                "run_duration_rel_error",
                "switches",
                "posterior_entropy",
            )
            result = {k: float(np.median([a[k] for a in group])) for k in metrics}
            if weight == 1:
                result["proper_unweighted_test_ll_per_row"] = float(
                    np.median([a["proper_unweighted_test_ll_per_row"] for a in group])
                )
            summary[f"{encoding}-{weight}"] = result
    payload = {
        "schema": "nhsmm-trading-variance-encoding-emission-weight-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seeds": [901, 902, 903],
        "window": 240,
        "train_windows": 8,
        "eval_windows": 4,
        "max_iter": 4,
        "transition_refine_steps": 2,
        "full_protocol": False,
        "weights_predeclared": list(WEIGHTS),
        "runs": rows,
        "summary": summary,
        "contract": {
            "weight": "inference-only log-emission multiplier; not a normalized density",
            "likelihood": "reported only at weight=1 under proper Gaussian likelihood",
            "labels": "scoring only; no training, weight selection or default promotion",
            "feature_dependence": "pooled training correlations; not proof of residual dependence",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
