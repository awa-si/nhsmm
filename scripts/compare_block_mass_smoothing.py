"""Checkpointed matched fixed-bin smoothing experiment; oracle OFF."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.adaptive_width_cap import validate_state
from scripts.block_mass_smoothing import geometry, marginal_objective, smooth_mstep
from scripts.compare_adaptive_large_selection import digest
from scripts.compare_adaptive_sample_size import save
from scripts.compare_adaptive_width_cap import SOURCES as PRIOR_SOURCES
from scripts.compare_adaptive_width_cap import evaluate as evaluate_paired
from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.verify_isolated_trading_hsmm import D, sequences

PROTOCOL = Path("docs/validation/block-mass-smoothing-protocol.json")
ORDER = ("warm", "unsmoothed", "smoothed")
SOURCES = tuple(
    dict.fromkeys(
        (
            *PRIOR_SOURCES,
            "scripts/block_mass_smoothing.py",
            "scripts/compare_block_mass_smoothing.py",
            str(PROTOCOL),
        )
    )
)


def expected_payload(seed, source_commit):
    contract = json.loads(PROTOCOL.read_text())
    if seed not in contract["seeds"]:
        raise ValueError("seed outside frozen protocol")
    return {
        "schema": "nhsmm-block-mass-smoothing-v1",
        "seed": seed,
        "execution": {
            "source_commit": source_commit,
            "runtime": "openai-native-workspace",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES},
        },
        "contract": contract,
        "phases": {},
        "active": None,
        "evaluation": None,
    }


def load_checkpoint(path, expected):
    r = json.loads(path.read_text())
    for key in ("schema", "seed", "execution", "contract"):
        if r.get(key) != expected[key]:
            raise ValueError(f"resume {key} differs")
    if list(r["phases"]) != list(ORDER[: len(r["phases"])]):
        raise ValueError("invalid completed prefix")
    if r["active"] is not None and (
        len(r["phases"]) == len(ORDER) or r["active"].get("key") != ORDER[len(r["phases"])]
    ):
        raise ValueError("invalid active phase")
    if r["evaluation"] is not None and (
        r["active"] is not None or list(r["phases"]) != list(ORDER)
    ):
        raise ValueError("premature evaluation")
    return r


def controls(method):
    if method["pseudocount_per_age"] != 1e-6 or method["solver_newton_polish_max_iterations"] != 3:
        raise ValueError("frozen solver method differs")
    return {
        "width": 8,
        "pseudocount": method["pseudocount_per_age"],
        "max_iterations": method["solver_max_iterations"],
        "max_evaluations": method["solver_max_evaluations"],
        "gradient_tolerance": 1e-8,
    }


def check_history(history, updates, complete=False):
    if (
        not history
        or [h["completed_updates"] for h in history] != list(range(len(history)))
        or len(history) > updates + 1
        or (complete and len(history) != updates + 1)
    ):
        raise ValueError("invalid history budget")
    if not all(
        np.isfinite(h[k]) for h in history for k in ("ll", "penalized_objective", "elapsed_seconds")
    ) or any(
        b["penalized_objective"] < a["penalized_objective"] - 1e-6
        for a, b in zip(history, history[1:])
    ):
        raise ValueError("invalid history objective")
    for h in history[1:]:
        e = h["solver"]
        if (
            not np.isfinite(e["maximum_normalized_anchored_gradient"])
            or not 0 <= e["maximum_normalized_anchored_gradient"] <= 1e-8
            or not 0 <= e["solver_iterations"] <= 1000
            or not 0 <= e["closure_evaluations"] <= 2000
            or len(e["q_start"]) != 4
            or len(e["q_final"]) != 4
            or not np.isfinite(e["q_start"] + e["q_final"]).all()
            or any(b < a - 1e-7 for a, b in zip(e["q_start"], e["q_final"], strict=True))
        ):
            raise ValueError("invalid solver certificate")


def fit_fixed(
    x,
    strength,
    *,
    iterations,
    solver_controls,
    initial_pmf=None,
    resume_history=None,
    checkpoint=None,
    tag="fixed",
):
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("nonnegative integer updates required")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("nonnegative finite strength required")
    blocks = geometry(D, 8)[0]
    parts = [blocks for _ in range(4)]
    p = (
        np.full((4, D), 1 / D)
        if initial_pmf is None
        else np.array(initial_pmf, dtype=float, copy=True)
    )
    validate_state(p, parts, 8)
    history = []
    first = 0
    offset = 0.0
    certificate = None
    if resume_history is not None:
        if initial_pmf is None:
            raise ValueError("resume requires saved PMF")
        check_history(resume_history, iterations)
        first = len(resume_history) - 1
        history = [dict(h) for h in resume_history[:-1]]
        offset = resume_history[-1]["elapsed_seconds"]
        certificate = resume_history[-1]["solver"]
    scores = emission_scores(x)
    start = time.monotonic()
    for update in range(first, iterations + 1):
        ll, counts = aggregate(scores, p)
        score = marginal_objective(ll, p, strength)
        if resume_history is not None and update == first:
            saved = resume_history[-1]
            if abs(ll - saved["ll"]) > 1e-7 or abs(score - saved["penalized_objective"]) > 1e-7:
                raise ValueError("resume PMF does not reproduce evidence/objective")
        history.append(
            {
                "completed_updates": update,
                "ll": ll,
                "penalized_objective": score,
                "elapsed_seconds": offset + time.monotonic() - start,
                "solver": certificate,
            }
        )
        if update and score < history[-2]["penalized_objective"] - 1e-6:
            raise AssertionError("penalized marginal objective decreased")
        if update == first or update % 5 == 0 or update == iterations:
            print(f"{tag} EM={update}/{iterations}", flush=True)
            if checkpoint is not None:
                checkpoint(p.copy(), [dict(h) for h in history])
        if update < iterations:
            p, certificate = smooth_mstep(counts, strength, initial_pmf=p, **solver_controls)
    return p, history


def train(r, x, path, max_phases=None):
    """Only training observations enter fitting; no labels/truth or outer data."""
    m = r["contract"]["method"]
    d = r["contract"]["data"]
    opts = controls(m)
    if x.shape != (d["training_sequences"], d["steps"], 1):
        raise ValueError("training dimensions differ")
    sha = digest(x)
    if "training_sha256" in r and r["training_sha256"] != sha:
        raise ValueError("training replay differs")
    r["training_sha256"] = sha
    used = 0
    warm = None
    blocks = geometry(D, 8)[0]
    for key in ORDER:
        completed = key in r["phases"]
        if not completed and max_phases is not None and used >= max_phases:
            save(path, r)
            return False
        updates = m["warm_updates"] if key == "warm" else m["penalized_updates_per_arm"]
        strength = float(np.log(len(x))) if key == "smoothed" else 0.0
        identity = {
            "updates": updates,
            "strength": strength,
            "training_sha256": sha,
            "blocks": [[list(b) for b in blocks] for _ in range(4)],
        }
        record = r["phases"].get(key) if completed else r["active"]
        if record is not None:
            if any(record.get(k) != v for k, v in identity.items()) or (
                not completed and record.get("key") != key
            ):
                raise ValueError("phase identity differs")
            check_history(record["history"], updates, completed)
            validate_state(record["pmf"], record["blocks"], 8)
        initial = record["pmf"] if record else warm

        def checkpoint(p, history):
            r["active"] = {"key": key, **identity, "pmf": p.tolist(), "history": history}
            save(path, r)

        p, h = fit_fixed(
            x,
            strength,
            iterations=updates,
            solver_controls=opts,
            initial_pmf=initial,
            resume_history=record["history"] if record else None,
            checkpoint=None if completed else checkpoint,
            tag=key,
        )
        if not completed:
            r["phases"][key] = {**identity, "pmf": p.tolist(), "history": h}
            r["active"] = None
            save(path, r)
            used += 1
        if key == "warm":
            warm = p
    return True


def evaluate(r, x):
    if r["active"] is not None or list(r["phases"]) != list(ORDER):
        raise ValueError("evaluation needs both completed fits")
    # Reuse the existing matched full-support evaluator and its numerical
    # owners. This immutable view only translates arm names, not data/methods.
    view = {
        **r,
        "phases": {
            "warm": r["phases"]["warm"],
            "unrestricted": r["phases"]["unsmoothed"],
            "width8": r["phases"]["smoothed"],
        },
    }
    e = evaluate_paired(view, x)
    return {
        "heldout_sha256": e["heldout_sha256"],
        "uniform_heldout_ll": e["uniform_heldout_ll"],
        "arms": {"unsmoothed": e["arms"]["unrestricted"], "smoothed": e["arms"]["width8"]},
        "smoothed_minus_unsmoothed_heldout_ll": e["width8_minus_unrestricted_heldout_ll"],
        "actual_msteps": e["actual_msteps"],
        "labels_used_in_learning_or_selection": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(1912, 1913, 1914))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-phases", type=int)
    args = parser.parse_args()
    if args.max_phases is not None and args.max_phases < 1:
        parser.error("positive max-phases required")
    r = expected_payload(args.seed, args.source_commit)
    if args.resume:
        r = load_checkpoint(args.output, r)
    elif args.output.exists():
        parser.error("preserve evidence: use --resume")
    if r["evaluation"] is not None:
        print("completed experiment; no new fitting or evaluation", flush=True)
        return
    torch.set_num_threads(1)
    d = r["contract"]["data"]
    x, _ = sequences(
        args.seed + d["training_seed_offset"],
        count=d["training_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if train(r, x, args.output, args.max_phases):
        r["evaluation"] = evaluate(r, x)
        save(args.output, r)
        print(json.dumps(r["evaluation"]), flush=True)
    else:
        print(f"phase boundary: completed {list(r['phases'])}", flush=True)


if __name__ == "__main__":
    main()
