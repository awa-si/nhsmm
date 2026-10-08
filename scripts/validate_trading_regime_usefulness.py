from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence

REGIMES = ("bull", "bear", "range", "chaotic")
K = 4
D = 64
FEATURES = (
    "log_return_1",
    "log_return_3",
    "log_return_12",
    "trend_12",
    "trend_48",
    "ewma_vol_12",
    "ewma_vol_48",
    "volume_z_48",
)
WINDOW = 480
TRAIN_WINDOWS = 24
EVAL_WINDOWS = 8
SEEDS = (901, 902, 903)


def _windows(data: dict[str, np.ndarray], start: int, end: int, count: int):
    usable = end - start - WINDOW
    anchors = np.linspace(start, start + usable, count, dtype=int)
    x = np.stack(
        [
            np.column_stack([data[name][a : a + WINDOW] for name in FEATURES])
            for a in anchors
        ]
    ).astype(np.float32)
    c = np.stack(
        [data["macro_stress"][a : a + WINDOW, None] for a in anchors]
    ).astype(np.float32)
    y = np.stack(
        [data["gt_regime_id"][a : a + WINDOW] for a in anchors]
    ).astype(np.int64)
    age = np.stack(
        [data["gt_episode_age"][a : a + WINDOW] for a in anchors]
    ).astype(np.int64)
    rem = np.stack(
        [data["gt_remaining_duration"][a : a + WINDOW] for a in anchors]
    ).astype(np.int64)
    gt_p = np.stack(
        [
            np.column_stack(
                [data[f"gt_p_next_{name}"][a : a + WINDOW] for name in REGIMES]
            )
            for a in anchors
        ]
    ).astype(np.float32)
    return x, c, y, age, rem, gt_p


def _standardize(train_x, *others):
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std, *[(x - mean) / std for x in others])


def _standardize_context(train_c, *others):
    mean = float(train_c.mean())
    std = float(train_c.std())
    if std < 1e-6:
        std = 1.0
    return ((train_c - mean) / std, *[(x - mean) / std for x in others])


def _best_perm(true, pred):
    best_acc = -1.0
    best = tuple(range(K))
    for perm in itertools.permutations(range(K)):
        mapped = np.asarray(perm, dtype=np.int64)[pred]
        acc = float(np.mean(mapped == true))
        if acc > best_acc:
            best_acc = acc
            best = perm
    return best, best_acc


def _ari(true, pred):
    cont = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(true, pred):
        cont[int(a), int(b)] += 1

    def c2(v):
        return v * (v - 1) // 2

    n = len(true)
    total = c2(n)
    sc = sum(c2(int(v)) for v in cont.ravel())
    sr = sum(c2(int(v)) for v in cont.sum(1))
    scol = sum(c2(int(v)) for v in cont.sum(0))
    expected = sr * scol / total
    upper = 0.5 * (sr + scol)
    den = upper - expected
    return float((sc - expected) / den) if abs(den) > 1e-12 else 0.0


def _boundary_f1(true, pred, tol=5):
    tp = fp = fn = 0
    for expected, actual in zip(true, pred):
        true_bounds = list(np.flatnonzero(expected[1:] != expected[:-1]) + 1)
        pred_bounds = list(np.flatnonzero(actual[1:] != actual[:-1]) + 1)
        used_true: set[int] = set()
        used_pred: set[int] = set()
        pairs = sorted(
            (abs(int(p) - int(t)), pi, ti)
            for pi, p in enumerate(pred_bounds)
            for ti, t in enumerate(true_bounds)
            if abs(int(p) - int(t)) <= tol
        )
        for _, pi, ti in pairs:
            if pi not in used_pred and ti not in used_true:
                used_pred.add(pi)
                used_true.add(ti)
        tp += len(used_true)
        fp += len(pred_bounds) - len(used_pred)
        fn += len(true_bounds) - len(used_true)
    den = 2 * tp + fp + fn
    return 1.0 if den == 0 else float(2 * tp / den)


def _run_medians(states):
    out = {i: [] for i in range(K)}
    for seq in states:
        start = 0
        while start < len(seq):
            end = start + 1
            while end < len(seq) and seq[end] == seq[start]:
                end += 1
            out[int(seq[start])].append(end - start)
            start = end
    return {i: (float(np.median(v)) if v else math.nan) for i, v in out.items()}


def _config(seed, *, context, hmm):
    return ModelConfig(
        n_states=K,
        n_features=len(FEATURES),
        max_duration=1 if hmm else D,
        duration_tail=not hmm,
        duration_tail_init_probability=0.5,
        causal=True,
        context_dim=1 if context else None,
        use_context_encoder=False,
        dropout=0.0,
        seed=seed,
        n_init=1,
        max_iter=30,
        lr=8e-3,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        emission_init_mode="kmeans",
        duration_init_mode="uniform",
        transition_init_mode="uniform",
        initial_init_mode="uniform",
        transition_context_max_delta=1.5,
        transition_refine_steps=12 if context else 0,
        transition_refine_lr=3e-2,
    )


def _fit(seed, x, c, *, context, hmm):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model = NHSMM(_config(seed, context=context, hmm=hmm), device="cpu")
    model.initialize_distributions()
    tx = torch.tensor(x)
    tc = torch.tensor(c) if context else None
    model.optimize(tx, context=tc)
    model.eval()
    return model


def _ll(model, x, c=None):
    with torch.inference_mode():
        value = model.log_likelihood(
            torch.tensor(x),
            context=None if c is None else torch.tensor(c),
            reduce=True,
        )
    return float(value) / float(x.shape[0] * x.shape[1])


def _gaussian_ll(train_x, test_x):
    flat = train_x.reshape(-1, train_x.shape[-1]).astype(np.float64)
    mu = flat.mean(0)
    var = np.maximum(flat.var(0), 1e-6)
    z = test_x.reshape(-1, test_x.shape[-1]).astype(np.float64)
    lp = -0.5 * (np.log(2 * np.pi * var) + ((z - mu) ** 2) / var)
    return float(lp.sum(1).mean())


def _aligned_transition_mae(model, context, true_state, gt_p, perm):
    inv = np.empty(K, dtype=np.int64)
    for learned, true in enumerate(perm):
        inv[true] = learned

    mask = np.zeros(true_state.shape, dtype=bool)
    mask[:, :-1] = true_state[:, 1:] != true_state[:, :-1]
    cc = context[mask]
    yy = true_state[mask]
    pp = gt_p[mask]
    if len(cc) == 0:
        return math.nan

    with torch.inference_mode():
        mats = (
            model.dist.transition.expected_probs(
                context=torch.tensor(cc[:, None, :], dtype=torch.float32)
            )
            .detach()
            .cpu()
            .numpy()[:, 0]
        )

    errors = []
    for mat, state, target in zip(mats, yy, pp):
        learned_row = mat[inv[int(state)]]
        aligned = np.empty(K, dtype=np.float64)
        for learned_target, true_target in enumerate(perm):
            aligned[true_target] = learned_row[learned_target]
        errors.append(np.mean(np.abs(aligned - target)))
    return float(np.mean(errors))


def _one(seed, path):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    loaded = np.load(path)
    data = {name: loaded[name] for name in loaded.files}
    n = len(data["close"])
    train_range = (0, int(0.60 * n))
    val_range = (int(0.60 * n), int(0.80 * n))
    test_range = (int(0.80 * n), n)

    train = _windows(data, *train_range, TRAIN_WINDOWS)
    validation = _windows(data, *val_range, EVAL_WINDOWS)
    test = _windows(data, *test_range, EVAL_WINDOWS)

    train_x, val_x, test_x = _standardize(train[0], validation[0], test[0])
    train_c, val_c, test_c = _standardize_context(
        train[1], validation[1], test[1]
    )

    full = _fit(seed, train_x, train_c, context=True, hmm=False)
    no_context = _fit(seed + 100, train_x, train_c, context=False, hmm=False)
    hmm = _fit(seed + 200, train_x, train_c, context=False, hmm=True)

    test_tensor = torch.tensor(test_x)
    context_tensor = torch.tensor(test_c)
    with torch.inference_mode():
        trace = filter_model_sequence(full, test_tensor, context=context_tensor)
        posterior = trace.state_posterior.detach().cpu().numpy()
        predicted = posterior.argmax(-1)

    true = test[2]
    perm, accuracy = _best_perm(true.reshape(-1), predicted.reshape(-1))
    mapped = np.asarray(perm, dtype=np.int64)[predicted]
    ari = _ari(true.reshape(-1), predicted.reshape(-1))
    boundary_f1 = _boundary_f1(true, mapped, tol=5)

    true_medians = _run_medians(true)
    predicted_medians = _run_medians(mapped)
    relative_errors = []
    for state in range(K):
        if (
            np.isfinite(true_medians[state])
            and np.isfinite(predicted_medians[state])
            and true_medians[state] > 0
        ):
            relative_errors.append(
                abs(predicted_medians[state] - true_medians[state])
                / true_medians[state]
            )
    duration_error = float(np.median(relative_errors))

    transition_mae = _aligned_transition_mae(full, test_c, true, test[5], perm)

    full_ll = _ll(full, test_x, test_c)
    no_context_ll = _ll(no_context, test_x)
    hmm_ll = _ll(hmm, test_x)
    gaussian_ll = _gaussian_ll(train_x, test_x)
    occupancy = posterior.reshape(-1, K).mean(0)

    return {
        "seed": seed,
        "matched_accuracy": accuracy,
        "ari": ari,
        "boundary_f1_tol5": boundary_f1,
        "duration_median_rel_error": duration_error,
        "transition_mae": transition_mae,
        "full_ll_per_row": full_ll,
        "hsmm_no_context_ll_per_row": no_context_ll,
        "hmm_state_only_ll_per_row": hmm_ll,
        "no_regime_gaussian_ll_per_row": gaussian_ll,
        "full_minus_hsmm_no_context": full_ll - no_context_ll,
        "full_minus_hmm": full_ll - hmm_ll,
        "full_minus_no_regime": full_ll - gaussian_ll,
        "occupancy": occupancy.tolist(),
        "true_run_medians": {REGIMES[i]: true_medians[i] for i in range(K)},
        "predicted_run_medians": {
            REGIMES[i]: predicted_medians[i] for i in range(K)
        },
        "permutation_learned_to_true": list(perm),
    }


def _worker(args):
    return _one(*args)


def _summary(rows):
    def med(key):
        return float(np.median([row[key] for row in rows]))

    result = {
        "n": len(rows),
        "median_matched_accuracy": med("matched_accuracy"),
        "median_ari": med("ari"),
        "median_boundary_f1_tol5": med("boundary_f1_tol5"),
        "median_duration_median_rel_error": med("duration_median_rel_error"),
        "median_transition_mae": med("transition_mae"),
        "median_full_ll_per_row": med("full_ll_per_row"),
        "median_hsmm_no_context_ll_per_row": med("hsmm_no_context_ll_per_row"),
        "median_hmm_state_only_ll_per_row": med("hmm_state_only_ll_per_row"),
        "median_no_regime_gaussian_ll_per_row": med("no_regime_gaussian_ll_per_row"),
        "median_full_minus_hsmm_no_context": med("full_minus_hsmm_no_context"),
        "median_full_minus_hmm": med("full_minus_hmm"),
        "median_full_minus_no_regime": med("full_minus_no_regime"),
    }
    checks = {
        "state_recovery": (
            result["median_matched_accuracy"] >= 0.65
            and result["median_ari"] >= 0.40
        ),
        "boundary_recovery": result["median_boundary_f1_tol5"] >= 0.45,
        "duration_recovery": result["median_duration_median_rel_error"] <= 0.50,
        "transition_recovery": result["median_transition_mae"] <= 0.20,
        "beats_no_regime": result["median_full_minus_no_regime"] >= 0.05,
        "not_worse_than_hmm": result["median_full_minus_hmm"] >= 0.0,
        "not_worse_than_no_context_hsmm": (
            result["median_full_minus_hsmm_no_context"] >= -0.001
        ),
    }
    checks["passed"] = all(checks.values())
    result["acceptance"] = checks
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch usefulness benchmark on the labeled trading-regime dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    jobs = [(seed, args.dataset) for seed in SEEDS]
    if args.workers == 1:
        rows = [_worker(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            rows = list(executor.map(_worker, jobs))

    summary = _summary(rows)
    payload = {
        "schema": "nhsmm-synthetic-trading-usefulness-v1",
        "dataset": str(args.dataset),
        "features": list(FEATURES),
        "context": "macro_stress",
        "k": K,
        "max_duration": D,
        "window": WINDOW,
        "train_windows": TRAIN_WINDOWS,
        "eval_windows": EVAL_WINDOWS,
        "seeds": list(SEEDS),
        "runs": rows,
        "summary": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
