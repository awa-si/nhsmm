"""Independent terminal/data/public numerical replay; no optimizer updates."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.historical_evidence import check_archived_bytes, historical_record
from scripts.compare_block_mass_smoothing import SOURCES
from scripts.compare_compact_duration_em import public_check
from scripts.compare_optimizer_bridge import PROTOCOL
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import make_model, sequences, true_law

SOURCE = "21428b6f0537944355fc5449259b3de4c31855bd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    if isinstance(actual, dict):
        require(actual.keys() == expected.keys(), f"{name}: keys")
        for key in actual:
            close(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"{name}: length")
        for a, b in zip(actual, expected, strict=True):
            close(a, b, name)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        require(
            np.isfinite(actual) and np.isfinite(expected) and abs(actual - expected) <= 1e-7,
            f"{name}: numerical mismatch",
        )
    else:
        require(actual == expected, f"{name}: mismatch")


def audit(path):
    """Audit a registered historical report and its sealed proof without replay.

    Recorded software is compared with the Git-anchored evidence index, never
    claimed to be the current interpreter. Numerical replay remains verify().
    """
    raw = path.read_bytes()
    report = json.loads(raw)
    frozen = historical_record("optimizer")
    require(report.get("schema") == frozen["schema"], "schema")
    require(
        report.get("complete") is True and report.get("contract") == frozen["contract"],
        "incomplete/changed contract",
    )
    identity = report["execution"]
    require(identity.get("source_commit") == SOURCE, "source commit")
    require(
        identity.get("source_sha256") == frozen["execution"]["source_sha256"], "source manifest"
    )
    require(identity == frozen["execution"], "recorded software/execution identity")
    require([row["seed"] for row in report["rows"]] == frozen["contract"]["seeds"], "seed prefix")
    for row in report["rows"]:
        require(list(row["arms"]) == frozen["contract"]["arms"], "arms")
        initial = frozen["initial_ll_by_seed"][str(row["seed"])]
        for key, arm in row["arms"].items():
            field = "ll" if key == "em" else "target_ll"
            close(arm["history"][0][field], initial, "recorded uniform start")
    proof = check_archived_bytes(raw, frozen["reports"]["pilot"])
    require(proof.get("verified") is True, "original verification certificate")
    require(proof.get("source_commit") == SOURCE, "original proof source")
    require(
        proof.get("input_sha256") == hashlib.sha256(raw).hexdigest(), "original proof input SHA"
    )
    require(proof.get("verification_optimizer_updates") == 0, "original verification updates")
    require(
        proof.get("independent_terminal_outer_public_replay") is True, "original replay certificate"
    )
    return {
        "schema": "nhsmm-optimizer-bridge-historical-audit-v1",
        "verified_historical_record": True,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "recorded_execution": identity,
        "recorded_summary": proof["summary"],
        "verification_optimizer_updates": 0,
        "independent_terminal_outer_public_replay": False,
        "scope": "read-only metadata/byte/certificate audit; no new numerical replay",
    }


def verify(path, *, historical=False):
    torch.set_num_threads(1)
    raw = path.read_bytes()
    report = json.loads(raw)
    contract = json.loads(PROTOCOL.read_text())
    require(report["schema"] == "nhsmm-optimizer-bridge-pilot-v1", "schema")
    require(
        report["contract"] == contract and report["complete"] is True, "incomplete/changed contract"
    )
    identity = report["execution"]
    require(identity["source_commit"] == SOURCE, "source commit")
    require(
        (identity["python"], identity["numpy"], identity["torch"])
        == (platform.python_version(), np.__version__, torch.__version__),
        "software",
    )
    paths = tuple(dict.fromkeys((*SOURCES, str(PROTOCOL), "scripts/compare_optimizer_bridge.py")))
    current = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    expected = current
    if historical:
        frozen = json.loads(Path("docs/validation/optimizer-bridge-frozen-source.json").read_text())
        require(frozen["source_commit"] == SOURCE, "historical source identity")
        expected = frozen["source_sha256"]
    require(identity["source_sha256"] == expected, "source manifest")
    require([row["seed"] for row in report["rows"]] == contract["seeds"], "seed prefix")
    summary = []
    for row in report["rows"]:
        seed = row["seed"]
        x, _ = sequences(seed + 400000, count=4, steps=60, sigma=2.0)
        outer, labels = sequences(seed + 500000, count=2, steps=60, sigma=2.0)
        require(
            row["training_sha256"] == hashlib.sha256(x.tobytes()).hexdigest(), "training identity"
        )
        require(
            row["outer_sha256"] == hashlib.sha256(outer.tobytes()).hexdigest(), "outer identity"
        )
        require(
            row["labels_used_in_learning_or_selection"] is False
            and row["numerical_checks_pass"] is True,
            "labels/numerical status",
        )
        require(list(row["arms"]) == contract["arms"], "arms")
        initial = np.full((4, 260), 1 / 260)
        baseline, _ = aggregate(emission_scores(outer), initial)
        initial_ll, _ = aggregate(emission_scores(x), initial)
        result = {"seed": seed, "arms": {}}
        for key, arm in row["arms"].items():
            p = np.asarray(arm["pmf"])
            require(
                p.shape == (4, 260) and np.isfinite(p).all() and (p > 0).all(), "full positive PMF"
            )
            require(np.max(np.abs(p.sum(1) - 1)) <= 1e-12, "PMF normalization")
            history = arm["history"]
            if key == "em":
                require([h["completed_updates"] for h in history] == [0, 1, 2], "EM budget")
                close(history[0]["ll"], initial_ll, "EM uniform start")
                terminal_ll, _ = aggregate(emission_scores(x), p)
                close(history[-1]["ll"], terminal_ll, "EM terminal")
                require(
                    all(b["ll"] >= a["ll"] - 1e-6 for a, b in zip(history, history[1:])),
                    "EM monotonicity",
                )
            else:
                require([h["completed_updates"] for h in history] == [0, 1], "Adam budget")
                require(arm["frozen_nuisance_preserved"] is True, "frozen certificate")
                close(history[0]["target_ll"], initial_ll, "Adam uniform start")
                for h in history:
                    i = h["completed_updates"]
                    temperature = (
                        1.0 if key == "constant" else 0.3 + 0.7 / (1 + np.exp(5 * (i - 1)))
                    )
                    close(h["scheduled_temperature"], temperature, "scheduled temperature")
                    close(h["target_temperature"], 1.0, "target temperature")
                    require(
                        np.isfinite(h["scheduled_ll"]) and np.isfinite(h["target_ll"]),
                        "history finite",
                    )
                model = make_model(p, sigma=2.0)
                with torch.no_grad():
                    public = float(model.log_likelihood(torch.tensor(x), reduce=True))
                terminal_ll, _ = aggregate(emission_scores(x), p)
                close(public, terminal_ll, "public training replay")
                close(arm["final_target_ll"], public, "restart/public target")
                require(0 <= arm["public_reference_ll_error"] <= 1e-7, "recorded public error")
            ll, _ = aggregate(emission_scores(outer), p)
            close(arm["heldout_ll"], ll, "outer LL")
            gain = (ll - baseline) / 120
            close(arm["gain_nats_per_row"], gain, "outer gain")
            measured = metrics(p, true_law())
            close(arm["metrics"], measured, "metrics")
            gradient = posterior_gradient_check(outer, p, support=np.ones_like(p, dtype=bool))
            filtering = public_check(p, outer, labels, support=np.ones_like(p, dtype=bool))
            close(arm["gradient_check"], gradient, "gradient replay")
            close(arm["public_recursion"], filtering, "filter replay")
            require(
                max(gradient.values()) <= 1e-7
                and filtering["ll_error"] <= 1e-7
                and filtering["maximum_state_probability_error"] <= 1e-8,
                "numerical gate",
            )
            result["arms"][key] = {
                "gain": gain,
                "pmf_tv": measured["pmf_tv"],
                "full_recovery_gate": max(measured["pmf_tv"]) <= 0.12 and gain > 0,
            }
        summary.append(result)
        print(f"seed{seed} independent terminal/outer replay passed", flush=True)
    noop = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_optimizer_bridge",
            "--source-commit",
            SOURCE,
            "--output",
            str(path),
            "--resume",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    source_matches_current = identity["source_sha256"] == current
    if source_matches_current:
        require(
            noop.returncode == 0
            and "completed pilot; no new fitting" in noop.stdout
            and path.read_bytes() == raw,
            "completed resume no-op",
        )
    else:
        require(
            historical
            and noop.returncode != 0
            and "source/software/contract drift" in noop.stderr
            and path.read_bytes() == raw,
            "historical resume must reject drift",
        )
    return {
        "schema": "nhsmm-optimizer-bridge-verification-v1",
        "verified": True,
        "source_commit": SOURCE,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files": len(paths),
        "actual_pilot_updates": 18,
        "verification_optimizer_updates": 0,
        "independent_terminal_outer_public_replay": True,
        "completed_resume_noop": source_matches_current,
        "historical_source_manifest": historical,
        "replay_source_matches_original": source_matches_current,
        "replay_source_sha256": current,
        "summary": summary,
        "scope": contract["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit registered historical bytes/certificates without replay",
    )
    args = parser.parse_args()
    if args.audit_only and (not args.historical or args.output.exists()):
        parser.error("--audit-only requires --historical and a new output path")
    proof = audit(args.input) if args.audit_only else verify(args.input, historical=args.historical)
    args.output.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
