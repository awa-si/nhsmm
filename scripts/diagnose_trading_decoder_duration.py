"""Compare causal filtering and retrospective MAP paths on identical short fits.

Labels enter scoring only. Learned episode durations are reported separately from
observable same-state run lengths, which can combine self-transition episodes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import filter_model_sequence
from scripts.validate_trading_regime_usefulness import (
    K,
    REGIMES,
    RunBudget,
    _ari,
    _best_perm,
    _boundary_f1,
    _duration_error,
    _fit,
    _ll,
    _run_medians,
    _standardize,
    _standardize_context,
    _windows,
)


def episode_median(probabilities, tail_hazard):
    """Median of masses 1..D-1 and a geometric D+ bucket."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) < 1 or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("probabilities must be a normalized vector")
    if not 0 < tail_hazard <= 1:
        raise ValueError("tail hazard must be in (0,1]")
    finite = np.cumsum(p[:-1])
    hits = np.flatnonzero(finite >= 0.5)
    if len(hits):
        return int(hits[0] + 1)
    remaining = (0.5 - float(finite[-1] if len(finite) else 0)) / p[-1]
    if tail_hazard == 1:
        return len(p)
    offsets = max(1, int(np.ceil(np.log1p(-remaining) / np.log1p(-tail_hazard))))
    return len(p) - 1 + offsets


def score(truth, predicted):
    perm, accuracy = _best_perm(truth.ravel(), predicted.ravel())
    mapped = np.asarray(perm)[predicted]
    medians = _run_medians(mapped)
    error, missing = _duration_error(_run_medians(truth), medians)
    return {
        "accuracy": accuracy,
        "ari": _ari(truth.ravel(), predicted.ravel()),
        "boundary_f1_tol5": _boundary_f1(truth, mapped),
        "run_duration_rel_error": error,
        "missing_states": missing,
        "run_medians": {REGIMES[i]: medians[i] for i in range(K)},
        "switches": int(np.count_nonzero(predicted[:, 1:] != predicted[:, :-1])),
        "permutation": list(perm),
    }


def complete_episode_medians(states, ages):
    """Exclude both window-edge episodes; include same-state episode resets."""
    durations = {i: [] for i in range(K)}
    for y, a in zip(states, ages):
        starts = np.flatnonzero(a == 1)
        for start, end in zip(starts[:-1], starts[1:]):
            durations[int(y[start])].append(int(end - start))
    return {
        REGIMES[i]: {
            "count": len(durations[i]),
            "median": float(np.median(durations[i])) if durations[i] else None,
        }
        for i in range(K)
    }


def diagnose(seed, path, budget, *, fit=_fit):
    torch.set_num_threads(1)
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    n = len(data["close"])
    train = _windows(data, 0, int(0.6 * n), budget.train_windows, window=budget.window)
    test = _windows(data, int(0.8 * n), n, budget.eval_windows, window=budget.window)
    train_x, test_x = _standardize(train[0], test[0])
    train_c, test_c = _standardize_context(train[1], test[1])
    model = fit(
        seed,
        train_x,
        train_c,
        context=True,
        hmm=False,
        emission_init_mode="kmeans_cluster_variance",
        budget=budget,
    )
    x, c = torch.tensor(test_x), torch.tensor(test_c)
    with torch.inference_mode():
        posterior = filter_model_sequence(model, x, context=c).state_posterior.numpy()
        filtered = posterior.argmax(-1)
        viterbi = np.stack([p.numpy() for p in model.predict(x, context=c, verbose=False)])
        # Use the same duration-bias and temperature as inference, at zero standardized context.
        lp = model.dist.duration.log_matrix(
            context=torch.zeros(1, 1, 1), temperature=1.0, T=1, soft_dmax=model.duration_logits_bias
        )
        probs = lp.exp().reshape(-1, K, model.config.max_duration)[0].numpy()
        tail = model.dist.duration.tail_probability().numpy().reshape(K)
    filter_score = score(test[2], filtered)
    duration = {}
    for learned, true in enumerate(filter_score["permutation"]):
        duration[REGIMES[true]] = {
            "episode_median": episode_median(probs[learned], float(tail[learned])),
            "d_plus_mass": float(probs[learned, -1]),
            "tail_end_hazard": float(tail[learned]),
        }
    return {
        "seed": seed,
        "train_ll_per_row": _ll(model, train_x, train_c),
        "test_ll_per_row": _ll(model, test_x, test_c),
        "filter": filter_score,
        "viterbi": score(test[2], viterbi),
        "decoder_agreement": float(np.mean(filtered == viterbi)),
        "true_switches": int(np.count_nonzero(test[2][:, 1:] != test[2][:, :-1])),
        "true_run_medians": {REGIMES[i]: v for i, v in _run_medians(test[2]).items()},
        "true_complete_episode_durations": complete_episode_medians(test[2], test[3]),
        "learned_episode_durations_at_zero_context": duration,
    }


def run(job):
    return diagnose(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end data required")
    budget = RunBudget(
        window=240, train_windows=8, eval_windows=4, max_iter=4, transition_refine_steps=2
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, [(s, args.dataset, budget) for s in (901, 902, 903)]))
    summary = {
        decoder: {
            metric: float(np.median([r[decoder][metric] for r in rows]))
            for metric in (
                "accuracy",
                "ari",
                "boundary_f1_tol5",
                "run_duration_rel_error",
                "switches",
            )
        }
        for decoder in ("filter", "viterbi")
    }
    payload = {
        "schema": "nhsmm-trading-decoder-duration-diagnostic-v1",
        "dataset": str(args.dataset),
        "dataset_schema": manifest["schema"],
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "budget": asdict(budget),
        "full_protocol": False,
        "runs": rows,
        "summary": summary,
        "interpretation": {
            "filter": "causal marginal argmax",
            "viterbi": "retrospective full-window MAP, unavailable for live decisions",
            "duration": "learned episode PMF differs from observable state runs with self-transitions",
            "labels": "scoring only; no model selection or supervised training",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
