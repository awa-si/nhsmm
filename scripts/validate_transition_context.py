from __future__ import annotations
import argparse
import concurrent.futures
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence

K = 3
F = 3
T = 240
N_TRAIN = 10
N_EVAL = 6
SEEDS = tuple(range(401, 416))
MEANS = np.eye(K, dtype=np.float32) * 4.0


@dataclass(frozen=True)
class Scenario:
    name: str
    p_hi: float
    p_lo: float
    min_acc: float | None = None
    min_ari: float | None = None
    max_mae: float | None = None
    min_corr: float | None = None
    min_dir: float | None = None
    max_null_delta: float | None = None
    min_noncollapsed: int = 14


SCENARIOS = {
    "strong": Scenario("strong", 0.85, 0.10, 0.90, 0.80, 0.15, 0.70, 0.80, None, 14),
    "moderate": Scenario("moderate", 0.60, 0.35, 0.75, 0.50, 0.18, 0.45, 0.70, None, 14),
    "null": Scenario("null", 0.475, 0.475, max_null_delta=0.08, min_noncollapsed=0),
}


def mats(s: Scenario):
    diag = max(0.0, 1.0 - s.p_hi - s.p_lo)
    p0 = np.array(
        [[diag, s.p_hi, s.p_lo], [s.p_lo, diag, s.p_hi], [s.p_hi, s.p_lo, diag]], dtype=np.float64
    )
    p1 = np.array(
        [[diag, s.p_lo, s.p_hi], [s.p_hi, diag, s.p_lo], [s.p_lo, s.p_hi, diag]], dtype=np.float64
    )
    return p0, p1


def gen(seed, s, n):
    rng = np.random.default_rng(seed)
    p0, p1 = mats(s)
    x = np.zeros((n, T, F), np.float32)
    z = np.zeros((n, T), np.int64)
    c = rng.integers(0, 2, size=(n, T, 1)).astype(np.float32)
    for b in range(n):
        z[b, 0] = rng.integers(K)
        for t in range(T):
            st = z[b, t]
            x[b, t] = MEANS[st] + rng.normal(0, 0.7, F).astype(np.float32)
            if t + 1 < T:
                z[b, t + 1] = rng.choice(K, p=(p1 if c[b, t, 0] > 0.5 else p0)[st])
    return x, z, c


def ari(true, pred):
    cont = np.zeros((K, K), np.int64)
    for a, b in zip(true, pred):
        cont[a, b] += 1

    def comb(v):
        return v * (v - 1) // 2

    n = len(true)
    total = comb(n)
    if total == 0:
        return 0.0
    sc = sum(comb(int(v)) for v in cont.ravel())
    sr = sum(comb(int(v)) for v in cont.sum(1))
    scol = sum(comb(int(v)) for v in cont.sum(0))
    ex = sr * scol / total
    up = 0.5 * (sr + scol)
    den = up - ex
    return float((sc - ex) / den) if abs(den) > 1e-12 else 0.0


def best_perm(true, pred):
    best = (-1, None)
    for perm in itertools.permutations(range(K)):
        mapped = np.fromiter((perm[s] for s in pred), dtype=np.int64, count=len(pred))
        a = float((mapped == true).mean())
        if a > best[0]:
            best = (a, perm)
    return best


def align_matrix(m, perm):
    # perm[learned] = true; invert to learned index for each true state
    inv = np.empty(K, np.int64)
    for learned, true in enumerate(perm):
        inv[true] = learned
    return m[np.ix_(inv, inv)]


def learned_mats(model):
    out = []
    with torch.inference_mode():
        for cv in (0.0, 1.0):
            ctx = torch.tensor([[[cv]]], dtype=torch.float32)
            p = model.dist.transition.expected_probs(context=ctx).detach().cpu().numpy()
            # [B,T,K,K]
            out.append(p.reshape(-1, K, K)[0].astype(np.float64))
    return out


def run(seed, s):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    tx, tz, tc = gen(seed, s, N_TRAIN)
    cfg = ModelConfig(
        n_states=K,
        n_features=F,
        max_duration=1,
        causal=True,
        context_dim=1,
        seed=seed,
        n_init=1,
        max_iter=40,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        emission_init_mode="kmeans",
        transition_type="ergodic",
        transition_context_max_delta=1.5,
        transition_refine_steps=20,
        transition_refine_lr=3e-2,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    model.optimize(torch.tensor(tx), context=torch.tensor(tc))
    model.eval()
    ex, ez, ec = gen(seed + 10000, s, N_EVAL)
    xt = torch.tensor(ex)
    ct = torch.tensor(ec)
    with torch.inference_mode():
        trace = filter_model_sequence(model, xt, context=ct)
        post = trace.state_posterior.cpu().numpy()
        pred = post.argmax(-1).reshape(-1)
        true = ez.reshape(-1)
        acc, perm = best_perm(true, pred)
        occupancy = post.reshape(-1, K).mean(0)
        occupancy /= occupancy.sum()
        clip = np.clip(occupancy, 1e-12, 1)
        eff = float(np.exp(-(clip * np.log(clip)).sum()))
        maxocc = float(occupancy.max())
        ll = float(model.log_likelihood(xt, context=ct, reduce=True)) / (N_EVAL * T)
        lm0, lm1 = learned_mats(model)
    lm0 = align_matrix(lm0, perm)
    lm1 = align_matrix(lm1, perm)
    tm0, tm1 = mats(s)
    mae = float((np.abs(lm0 - tm0).mean() + np.abs(lm1 - tm1).mean()) / 2)
    eps = 1e-9
    kl = float(
        (
            (tm0 * np.log((tm0 + eps) / (lm0 + eps))).sum(1).mean()
            + (tm1 * np.log((tm1 + eps) / (lm1 + eps))).sum(1).mean()
        )
        / 2
    )
    td = (tm1 - tm0).reshape(-1)
    ld = (lm1 - lm0).reshape(-1)
    active = np.abs(td) > 1e-8
    corr = (
        float(np.corrcoef(td[active], ld[active])[0, 1])
        if active.sum() >= 2 and np.std(ld[active]) > 1e-12
        else 0.0
    )
    direction = float((np.sign(td[active]) == np.sign(ld[active])).mean()) if active.any() else None
    null_delta = float(np.abs(ld).mean())
    return dict(
        seed=seed,
        scenario=s.name,
        matched_accuracy=acc,
        ari=ari(true, pred),
        effective_states=eff,
        max_occupancy=maxocc,
        noncollapsed=bool(eff >= 2.0 and maxocc <= 0.80),
        transition_mae=mae,
        transition_kl=kl,
        delta_correlation=corr,
        direction_accuracy=direction,
        null_abs_delta=null_delta,
        ll_per_row=ll,
        true_p0=tm0.tolist(),
        true_p1=tm1.tolist(),
        learned_p0=lm0.tolist(),
        learned_p1=lm1.tolist(),
    )


def summary(s, rows):
    def med(k):
        return float(np.median([r[k] for r in rows]))

    nc = sum(r["noncollapsed"] for r in rows)
    reasons = []
    vals = dict(
        median_accuracy=med("matched_accuracy"),
        median_ari=med("ari"),
        median_mae=med("transition_mae"),
        median_kl=med("transition_kl"),
        median_corr=med("delta_correlation"),
        median_null_abs_delta=med("null_abs_delta"),
        noncollapsed=nc,
    )
    dirs = [r["direction_accuracy"] for r in rows if r["direction_accuracy"] is not None]
    vals["median_direction"] = float(np.median(dirs)) if dirs else None
    if s.min_acc is not None and vals["median_accuracy"] < s.min_acc:
        reasons.append("accuracy")
    if s.min_ari is not None and vals["median_ari"] < s.min_ari:
        reasons.append("ari")
    if s.max_mae is not None and vals["median_mae"] > s.max_mae:
        reasons.append("mae")
    if s.min_corr is not None and vals["median_corr"] < s.min_corr:
        reasons.append("delta_correlation")
    if s.min_dir is not None and vals["median_direction"] < s.min_dir:
        reasons.append("direction")
    if s.max_null_delta is not None and vals["median_null_abs_delta"] > s.max_null_delta:
        reasons.append("null_delta")
    if nc < s.min_noncollapsed:
        reasons.append("noncollapsed")
    return dict(scenario=s.name, n=len(rows), **vals, passed=not reasons, failure_reasons=reasons)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _worker(job: tuple[int, Scenario]) -> dict[str, object]:
    return run(*job)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate context-conditioned transition recovery."
    )
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--seeds", type=_parse_seeds, default=SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selected: Iterable[Scenario] = (
        SCENARIOS.values() if args.scenario == "all" else (SCENARIOS[args.scenario],)
    )
    payload: dict[str, object] = {
        "schema": "nhsmm-transition-context-recovery-v2",
        "transition_context_max_delta": 1.5,
        "transition_refine_steps": 20,
        "transition_refine_lr": 3e-2,
        "seeds": list(args.seeds),
        "scenarios": {},
    }
    all_passed = True
    for scenario in selected:
        jobs = [(seed, scenario) for seed in args.seeds]
        if args.workers == 1:
            rows = [_worker(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(_worker, jobs))
        rows.sort(key=lambda row: int(row["seed"]))
        sm = summary(scenario, rows)
        payload["scenarios"][scenario.name] = {"summary": sm, "runs": rows}
        all_passed = all_passed and bool(sm["passed"])
        print(json.dumps(sm, sort_keys=True), flush=True)

    payload["passed"] = all_passed
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
