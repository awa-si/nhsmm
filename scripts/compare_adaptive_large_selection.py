"""Checkpointed training-only large-exposure selection; research, oracle OFF."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.compare_adaptive_duration_em import PSEUDOCOUNT, fit_adaptive
from scripts.compare_adaptive_sample_size import save
from scripts.compare_compact_duration_em import public_check
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import D, sequences, true_law

PROTOCOL = Path("docs/validation/adaptive-large-selection-protocol.json")
SOURCES = (
    "scripts/compare_adaptive_large_selection.py",
    "scripts/compare_adaptive_duration_em.py",
    "scripts/compare_adaptive_sample_size.py",
    "scripts/compare_compact_duration_em.py",
    "scripts/compare_unlabelled_duration_em.py",
    "scripts/verify_isolated_trading_hsmm.py",
    "scripts/verify_unlabelled_trading_hsmm.py",
    "scripts/diagnose_trading_ground_truth.py",
    "scripts/generate_trading_regime_dataset.py",
    "nhsmm/models/base.py",
    "nhsmm/filtering.py",
    "nhsmm/distributions/default.py",
    str(PROTOCOL),
)


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def expected_payload(protocol, seed, base_commit):
    if seed not in protocol["seeds"]:
        raise ValueError("seed is outside the frozen protocol")
    return {
        "schema": "nhsmm-adaptive-large-selection-v1",
        "execution": {
            "base_commit": base_commit,
            "runtime": "openai-native-workspace",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "source_sha256": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in SOURCES},
        },
        "contract": protocol,
        "seed": seed,
        "phases": {},
        "active": None,
        "selection": None,
        "evaluation": None,
    }


def load_checkpoint(path, expected):
    saved = json.loads(path.read_text())
    for key in ("schema", "execution", "contract", "seed"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"resume {key} differs from frozen experiment")
    return saved


class PhaseLimit(Exception):
    """A verified phase boundary, not an experiment failure."""


def train(payload, x, path, *, max_phases=None):
    """No outer observations or ground truth are accessible from this function."""
    c, seed = payload["contract"], payload["seed"]
    data, method = c["data"], c["method"]
    if x.shape[0] != data["training_sequences"] or method["pseudocount"] != PSEUDOCOUNT:
        raise ValueError("training dimensions or pseudocount differ")
    indices = np.random.default_rng(seed + data["split_seed_offset"]).permutation(len(x))
    ti, vi = indices[: data["inner_fit_sequences"]], indices[data["inner_fit_sequences"] :]
    if len(vi) != data["inner_validation_sequences"] or not len(ti) or not len(vi):
        raise ValueError("invalid inner split")
    split = {
        "train_indices": ti.tolist(),
        "validation_indices": vi.tolist(),
        "training_sha256": digest(x),
        "inner_fit_sha256": digest(x[ti]),
        "inner_validation_sha256": digest(x[vi]),
    }
    if "split" in payload and payload["split"] != split:
        raise ValueError("training or inner split replay differs")
    payload["split"] = split
    phases, used, visited = payload["phases"], 0, []
    ordered = ["inner_warm"] + [f"candidate_{i}" for i in range(len(method["factors"]))]
    ordered += ["full_warm", "selected_refit", "factor2_control"]
    if list(phases) != ordered[: len(phases)]:
        raise ValueError("completed phases must be an ordered prefix")
    validation_scores = emission_scores(x[vi])

    def fit(key, observations, penalty, updates, initial=None):
        nonlocal used
        visited.append(key)
        identity = {
            "training_sha256": digest(observations),
            "penalty": penalty,
            "updates": updates,
        }
        completed = key in phases
        active = payload["active"]
        if active is not None and active["key"] != key and not completed:
            raise ValueError("active phase differs from next incomplete phase")
        record = phases[key] if completed else active
        if record is not None:
            if any(record.get(k) != v for k, v in identity.items()):
                raise ValueError("phase data or budget differs")
            h = record["history"]
            if (
                not h
                or [r["completed_updates"] for r in h] != list(range(len(h)))
                or len(h) > updates + 1
                or (completed and len(h) != updates + 1)
            ):
                raise ValueError("invalid phase history")
            if not all(
                np.isfinite(r[k])
                for r in h
                for k in ("ll", "penalized_objective", "elapsed_seconds")
            ) or any(
                b["penalized_objective"] < a["penalized_objective"] - 1e-6 for a, b in zip(h, h[1:])
            ):
                raise ValueError("invalid phase objective evidence")
            if h[-1]["blocks_by_state"] != [len(b) for b in record["blocks"]]:
                raise ValueError("phase partition evidence differs")
        if not completed and max_phases is not None and used >= max_phases:
            if set(phases) != set(visited[:-1]):
                raise ValueError("completed phases are not the expected prefix")
            save(path, payload)
            raise PhaseLimit

        def checkpoint(p, blocks, history):
            payload["active"] = {
                "key": key,
                **identity,
                "pmf": p.tolist(),
                "blocks": blocks,
                "history": history,
            }
            save(path, payload)

        # Completed phases re-score once; the existing fitter verifies PMF,
        # partitions, update indices and final objective without another M-step.
        p, blocks, history = fit_adaptive(
            observations,
            penalty,
            iterations=updates,
            initial_state=(record["pmf"], record["blocks"]) if record else initial,
            resume_history=record["history"] if record else None,
            checkpoint=None if completed else checkpoint,
            tag=key,
        )
        if not completed:
            phases[key] = {
                **identity,
                "pmf": p.tolist(),
                "blocks": blocks,
                "history": history,
            }
            payload["active"] = None
            save(path, payload)
            used += 1
        return p, blocks

    try:
        inner_warm = fit("inner_warm", x[ti], 0.0, method["warm_updates"])
        candidates = []
        for i, factor in enumerate(method["factors"]):
            p, _ = fit(
                f"candidate_{i}",
                x[ti],
                float(factor * np.log(len(ti))),
                method["penalized_updates"],
                inner_warm,
            )
            ll, _ = aggregate(validation_scores, p)
            candidates.append({"factor": factor, "validation_ll": ll, "phase": f"candidate_{i}"})
        chosen = max(candidates, key=lambda r: (r["validation_ll"], r["factor"]))["factor"]
        selection = {"candidates": candidates, "selected_factor": chosen}
        if payload["selection"] is not None:
            old = payload["selection"]
            if old["selected_factor"] != chosen or any(
                abs(a["validation_ll"] - b["validation_ll"]) > 1e-7
                or a["factor"] != b["factor"]
                or a["phase"] != b["phase"]
                for a, b in zip(old["candidates"], candidates, strict=True)
            ):
                raise ValueError("selection replay differs")
        payload["selection"] = selection
        save(path, payload)  # Selection is durable before any outer data exists.
        full_warm = fit("full_warm", x, 0.0, method["warm_updates"])
        fit(
            "selected_refit",
            x,
            float(chosen * np.log(len(x))),
            method["penalized_updates"],
            full_warm,
        )
        if chosen != c["controls"]["frozen_factor"]:
            fit(
                "factor2_control",
                x,
                float(c["controls"]["frozen_factor"] * np.log(len(x))),
                method["penalized_updates"],
                full_warm,
            )
        if set(phases) != set(visited) or payload["active"] is not None:
            raise ValueError("unexpected completed or active phase")
    except PhaseLimit:
        return False
    return True


def evaluate(payload, x):
    """Outer data and truth exist only after training has completed."""
    if (
        payload["active"] is not None
        or payload["selection"] is None
        or "selected_refit" not in payload["phases"]
        or (
            payload["selection"]["selected_factor"]
            != payload["contract"]["controls"]["frozen_factor"]
            and "factor2_control" not in payload["phases"]
        )
    ):
        raise ValueError("outer evaluation requires completed selected and control fits")
    c, seed = payload["contract"], payload["seed"]
    data = c["data"]
    y, labels = sequences(
        seed + data["outer_seed_offset"],
        count=data["outer_sequences"],
        steps=data["steps"],
        sigma=data["sigma"],
    )
    support = np.ones((4, D), dtype=bool)
    scores = emission_scores(y)
    baseline, _ = aggregate(scores, support / D)
    truth = true_law()
    keys = {"selected": "selected_refit"}
    keys["factor2_control"] = (
        "factor2_control" if "factor2_control" in payload["phases"] else "selected_refit"
    )
    results = {}
    for key in dict.fromkeys(keys.values()):
        p = np.asarray(payload["phases"][key]["pmf"])
        ll, _ = aggregate(scores, p)
        m = metrics(p, truth)
        gain = (ll - baseline) / labels.size
        results[key] = {
            "metrics": m,
            "heldout_ll": ll,
            "heldout_gain_nats_per_row": gain,
            "full_gate": bool(gain > 0 and m["fine_pmf_gate"]),
            "gradient_check": posterior_gradient_check(x[:1], p, support=support),
            "public_recursion": public_check(p, y, labels, support=support),
        }
    return {
        "heldout_sha256": digest(y),
        "uniform_heldout_ll": baseline,
        "arm_phase_keys": keys,
        "results_by_phase": results,
        "labels_used_in_learning_or_selection": False,
        "actual_msteps": sum(p["updates"] for p in payload["phases"].values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(1906, 1907, 1908))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-phases", type=int, help="bound new completed phases per invocation")
    args = parser.parse_args()
    if args.max_phases is not None and args.max_phases < 1:
        parser.error("--max-phases must be positive")
    protocol = json.loads(PROTOCOL.read_text())
    payload = expected_payload(protocol, args.seed, args.base_commit)
    if args.resume:
        payload = load_checkpoint(args.output, payload)
    elif args.output.exists():
        parser.error("preserve existing evidence: use --resume or a new output path")
    if payload["evaluation"] is not None:
        print("completed experiment; no new fitting or evaluation", flush=True)
        return
    torch.set_num_threads(1)
    data = protocol["data"]
    x, _ = sequences(
        args.seed + data["training_seed_offset"],
        count=data["training_sequences"],
        steps=data["steps"],
        sigma=data["sigma"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if train(payload, x, args.output, max_phases=args.max_phases):
        payload["evaluation"] = evaluate(payload, x)
        save(args.output, payload)
        print(json.dumps(payload["evaluation"]), flush=True)
    else:
        print(f"phase boundary: completed {list(payload['phases'])}", flush=True)


if __name__ == "__main__":
    main()
