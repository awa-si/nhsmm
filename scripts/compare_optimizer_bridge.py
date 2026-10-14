"""Frozen small execution pilot, not statistical or joint-neural acceptance."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.compare_adaptive_sample_size import save
from scripts.compare_block_mass_smoothing import SOURCES
from scripts.compare_compact_duration_em import public_check
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    fit_em,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import D, make_model, model_law, sequences, true_law

PROTOCOL = Path("docs/validation/optimizer-bridge-protocol.json")


def run_seed(seed, contract):
    torch.set_num_threads(1)
    x, _ = sequences(
        seed + 400000,
        count=contract["training_sequences"],
        steps=contract["steps"],
        sigma=contract["sigma"],
    )
    initial = np.full((4, D), 1 / D)
    fitted = {}
    for arm in contract["arms"]:
        if arm == "em":
            p, history = fit_em(
                x, initial, iterations=contract["updates_per_arm"], label=f"seed{seed} EM"
            )
            fitted[arm] = {"pmf": p.tolist(), "history": history}
            continue
        model = make_model(initial, sigma=contract["sigma"])
        model.dist.duration.logits.requires_grad_(True)
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        cfg = model.config.with_overrides(
            n_init=1,
            max_iter=contract["updates_per_arm"],
            lr=contract["adam_lr"],
            optimizer_temperature_mode=arm,
            optimizer_target_temperature=1.0,
            optimizer_initialization="current",
            use_scheduler=False,
            convergence_stop=False,
            verbose=False,
            loss_bias=0.0,
        )
        original = model.dist
        model.optimize(torch.tensor(x), cfg=cfg)
        if model.dist is not original or any(
            not torch.equal(frozen[n], p)
            for n, p in model.named_parameters()
            if not p.requires_grad
        ):
            raise ValueError("prepared/frozen nuisance parameters changed")
        p = model_law(model).detach().numpy()
        with torch.no_grad():
            public = float(model.log_likelihood(torch.tensor(x), reduce=True))
        independent, _ = aggregate(emission_scores(x), p)
        if max(abs(public - independent), abs(public - model._restart_scores[0])) > 1e-7:
            raise ValueError("terminal target/public/reference likelihood mismatch")
        fitted[arm] = {
            "pmf": p.tolist(),
            "history": model._optimization_history[0],
            "final_target_ll": model._restart_scores[0],
            "public_reference_ll_error": abs(public - independent),
            "frozen_nuisance_preserved": True,
        }
    # Held-out observations, labels and true-law metrics are post-fit only.
    outer, labels = sequences(
        seed + 500000,
        count=contract["outer_sequences"],
        steps=contract["steps"],
        sigma=contract["sigma"],
    )
    baseline, _ = aggregate(emission_scores(outer), initial)
    for arm, row in fitted.items():
        p = np.asarray(row["pmf"])
        ll, _ = aggregate(emission_scores(outer), p)
        gradient = posterior_gradient_check(outer, p, support=np.ones_like(p, dtype=bool))
        filtering = public_check(p, outer, labels, support=np.ones_like(p, dtype=bool))
        if (
            max(gradient.values()) > 1e-7
            or filtering["ll_error"] > 1e-7
            or filtering["maximum_state_probability_error"] > 1e-8
        ):
            raise ValueError("public numerical gate")
        row.update(
            heldout_ll=ll,
            gain_nats_per_row=(ll - baseline) / outer.shape[0] / outer.shape[1],
            metrics=metrics(p, true_law()),
            gradient_check=gradient,
            public_recursion=filtering,
        )
    return {
        "seed": seed,
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "outer_sha256": hashlib.sha256(outer.tobytes()).hexdigest(),
        "arms": fitted,
        "labels_used_in_learning_or_selection": False,
        "numerical_checks_pass": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    contract = json.loads(PROTOCOL.read_text())
    paths = tuple(dict.fromkeys((*SOURCES, str(PROTOCOL), "scripts/compare_optimizer_bridge.py")))
    identity = {
        "source_commit": args.source_commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
    }
    report = {
        "schema": "nhsmm-optimizer-bridge-pilot-v1",
        "execution": identity,
        "contract": contract,
        "rows": [],
        "complete": False,
    }
    if args.output.exists():
        if not args.resume:
            raise ValueError("existing report requires explicit resume")
        report = json.loads(args.output.read_text())
        if report["execution"] != identity or report["contract"] != contract:
            raise ValueError("source/software/contract drift")
        if [r["seed"] for r in report["rows"]] != contract["seeds"][: len(report["rows"])]:
            raise ValueError("invalid completed seed prefix")
        if report["complete"]:
            if len(report["rows"]) != len(contract["seeds"]):
                raise ValueError("premature completion")
            print("completed pilot; no new fitting")
            return
    save(args.output, report)
    for seed in contract["seeds"][len(report["rows"]) :]:
        report["rows"].append(run_seed(seed, contract))
        save(args.output, report)
        print(f"seed{seed} verified numerical bridge", flush=True)
    report["complete"] = True
    save(args.output, report)
    print(json.dumps({"complete": True, "seeds": contract["seeds"], "scope": contract["scope"]}))


if __name__ == "__main__":
    main()
