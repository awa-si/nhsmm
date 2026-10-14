"""Freeze/integrity and independent terminal/public/outer replay, no fitting."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.compare_joint_static_chain import (
    digest,
    expected_payload,
    initial_states,
    load_checkpoint,
)
from scripts.static_chain_evaluation import evaluate
from scripts.verify_fixed_bin_optimizers import equal
from scripts.verify_isolated_trading_hsmm import sequences


def inspect(path, source, *, partial=False):
    seed = json.loads(path.read_text())["seed"]
    r = load_checkpoint(path, expected_payload(seed, source))
    d = r["contract"]["data"]
    x, _ = sequences(
        seed + d["training_seed_offset"],
        count=d["training_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    if digest(x) != r["training_sha256"]:
        raise ValueError("training hash")
    initial = initial_states(x)
    records = list(r["phases"].values()) + ([r["active"]] if r["active"] else [])
    for record in records:
        if record["initial"] != initial[record["key"]]:
            raise ValueError("training-only initialization mismatch")
    if partial:
        if r["evaluation"] is not None:
            raise ValueError("snapshot already evaluated")
    elif r["evaluation"] is None:
        raise ValueError("incomplete final report")
    return r, x


def verify(path, source):
    torch.set_num_threads(1)
    raw = path.read_bytes()
    r, x = inspect(path, source)
    result = evaluate(r, x)
    equal(result, r["evaluation"])
    if result["actual_em_updates"] != 90 or not all(
        v["numerical_gate"] for v in result["arms"].values()
    ):
        raise ValueError("budget/numerical gate")
    noop = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_joint_static_chain",
            "--seed",
            str(r["seed"]),
            "--source-commit",
            source,
            "--output",
            str(path),
            "--resume",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if (
        noop.returncode
        or "completed experiment; no new fitting or evaluation" not in noop.stdout
        or path.read_bytes() != raw
    ):
        raise ValueError("completed byte-preserving no-op")
    return {
        "schema": "nhsmm-joint-static-chain-verification-v1",
        "verified": True,
        "seed": r["seed"],
        "source_commit": source,
        "input_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files": len(r["execution"]["source_sha256"]),
        "verification_em_updates": 0,
        "source_software_contract_data_initialization_history_grid_pass": True,
        "independent_terminal_outer_public_replay_pass": True,
        "completed_resume_noop": True,
        "evaluation": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        p.error("preserve evidence: output already exists")
    if a.snapshot:
        raw = a.input.read_bytes()
        # Snapshot immutable bytes first to avoid joining different live atomic states.
        a.output.write_bytes(raw)
        try:
            r, _ = inspect(a.output, a.source_commit, partial=True)
        except Exception:
            a.output.unlink()
            raise
        print(
            json.dumps(
                {
                    "snapshot_verified": True,
                    "seed": r["seed"],
                    "phase": r["active"]["key"] if r["active"] else "completed fits",
                    "updates": (
                        len(r["active"]["history"]) - 1
                        if r["active"]
                        else sum(len(v["history"]) - 1 for v in r["phases"].values())
                    ),
                }
            ),
            flush=True,
        )
    else:
        result = verify(a.input, a.source_commit)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "verified": True,
                    "seed": result["seed"],
                    "arms": {
                        k: {
                            "tv": v["metrics"]["pmf_tv"],
                            "gain": v["heldout_gain_nats_per_row"],
                            "transition_tv": v["transition_row_tv"],
                            "initial_tv": v["initial_tv"],
                            "component_gates": v["component_gates"],
                            "gate": v["full_gate"],
                        }
                        for k, v in result["evaluation"]["arms"].items()
                    },
                    "paired_ll": result["evaluation"]["joint_minus_frozen_heldout_ll"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
