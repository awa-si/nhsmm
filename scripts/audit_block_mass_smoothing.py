"""Read-only audit of SHA-bound completed smoothing evidence; no fitting.

This audits recorded independent replay certificates. It does not itself repeat
terminal E-steps or outer likelihood calculations, and is not a signature check.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.historical_evidence import check_archived_bytes, historical_record
from scripts.adaptive_width_cap import validate_state
from scripts.block_mass_smoothing import geometry, marginal_objective
from scripts.compare_block_mass_smoothing import ORDER, check_history, expected_payload

SOURCE = "140ff55271b84b11798be5bd36afafa1ebc77a34"
SEEDS = (1912, 1913, 1914)
PREFIX = "block-mass-smoothing-"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(not isinstance(value, bool), f"{name}: boolean is not a numeric result")
    value = float(value)
    require(np.isfinite(value), f"{name}: nonfinite")
    return value


def audit_seed(raw, proof, seed, *, historical=False):
    """Validate one immutable report and its existing independent replay proof."""
    report = json.loads(raw)
    if historical:
        frozen = historical_record("smoothing")
        require(frozen["execution"]["source_commit"] == SOURCE, "frozen manifest identity")
        require(str(seed) in frozen["reports"], "seed outside frozen protocol")
        expected = {
            "schema": frozen["schema"],
            "seed": seed,
            "execution": frozen["execution"],
            "contract": frozen["contract"],
        }
    else:
        expected = expected_payload(seed, SOURCE)
    for key in ("schema", "seed", "execution", "contract"):
        require(report.get(key) == expected[key], f"seed{seed}: {key} differs")
    require(report["active"] is None, "unfinished active phase")
    require(list(report["phases"]) == list(ORDER), "incomplete or reordered phases")
    require(
        proof.get("schema") == "nhsmm-block-mass-smoothing-final-verification-v1", "proof schema"
    )
    require(proof.get("seed") == seed and proof.get("source_commit") == SOURCE, "proof identity")
    require(
        proof.get("source_hashes") == len(expected["execution"]["source_sha256"]), "source count"
    )
    require(proof.get("input_report_sha256") == hashlib.sha256(raw).hexdigest(), "raw SHA mismatch")
    for key in (
        "verified",
        "source_contract_software_training_outer_replayed",
        "matched_initial_pmf_partitions_budgets",
        "terminal_fits_replayed_without_msteps",
        "all_solver_certificates_pass",
        "numerical_checks_pass",
        "completed_resume_noop",
    ):
        require(proof.get(key) is True, f"missing replay certificate: {key}")
    blocks = [[list(b) for b in geometry(260, 8)[0]] for _ in range(4)]
    for key in ORDER:
        phase = report["phases"][key]
        updates = 3 if key == "warm" else 27
        strength = float(np.log(1024)) if key == "smoothed" else 0.0
        require(
            phase["updates"] == updates and phase["strength"] == strength, "phase budget/strength"
        )
        require(phase["training_sha256"] == report["training_sha256"], "phase data identity")
        require(phase["blocks"] == blocks, "fixed grid differs")
        validate_state(phase["pmf"], phase["blocks"], 8)
        check_history(phase["history"], updates, complete=True)
        if key != "warm":
            warm = report["phases"]["warm"]
            ll = warm["history"][-1]["ll"]
            require(abs(phase["history"][0]["ll"] - ll) <= 1e-7, "shared warm likelihood")
            objective = marginal_objective(ll, np.asarray(warm["pmf"]), strength)
            require(
                abs(phase["history"][0]["penalized_objective"] - objective) <= 1e-7,
                "shared warm objective",
            )
    evaluation = report["evaluation"]
    require(
        evaluation is not None and proof["evaluation"] == evaluation, "replay evaluation mismatch"
    )
    require(evaluation["actual_msteps"] == 57, "actual M-step budget")
    require(evaluation["labels_used_in_learning_or_selection"] is False, "oracle learning enabled")
    require(set(evaluation["arms"]) == {"unsmoothed", "smoothed"}, "arm identity")
    baseline = finite(evaluation["uniform_heldout_ll"], "baseline")
    for key, arm in evaluation["arms"].items():
        tv = [finite(v, "TV") for v in arm["metrics"]["pmf_tv"]]
        require(len(tv) == 4 and all(0 <= v <= 1 for v in tv), "invalid state TV")
        gain = finite(arm["heldout_gain_nats_per_row"], "gain")
        ll = finite(arm["heldout_ll"], "heldout LL")
        require(abs(gain - (ll - baseline) / 7200) <= 1e-12, "gain arithmetic")
        require(arm["full_gate"] is (max(tv) <= 0.12 and gain > 0), "full gate arithmetic")
        require(arm["metrics"]["fine_pmf_gate"] is (max(tv) <= 0.12), "PMF gate arithmetic")
        require(arm["blocks"] == [33] * 4, "arm grid summary")
        for value in arm["gradient_check"].values():
            require(0 <= finite(value, "gradient error") <= 1e-7, "gradient/LL numerical gate")
        for name, limit in (("ll_error", 1e-7), ("maximum_state_probability_error", 1e-8)):
            require(
                0 <= finite(arm["public_recursion"][name], name) <= limit, "filter numerical gate"
            )
    difference = finite(evaluation["smoothed_minus_unsmoothed_heldout_ll"], "paired LL")
    arms = evaluation["arms"]
    require(
        abs(difference - (arms["smoothed"]["heldout_ll"] - arms["unsmoothed"]["heldout_ll"]))
        <= 1e-7,
        "paired LL arithmetic",
    )
    if historical:
        check_archived_bytes(raw, frozen["reports"][str(seed)], proof=proof)
    return report


def audit_batch(directory, *, historical=False):
    reports = []
    hashes = {}
    for seed in SEEDS:
        path = directory / f"{PREFIX}seed{seed}-native-20261014.json"
        raw = path.read_bytes()
        proof = json.loads(
            (directory / f"{PREFIX}seed{seed}-verification-native-20261014.json").read_text()
        )
        reports.append(audit_seed(raw, proof, seed, historical=historical))
        hashes[str(seed)] = hashlib.sha256(raw).hexdigest()
    for field in ("training_sha256",):
        require(len({r[field] for r in reports}) == 3, "reused training data")
    require(len({r["evaluation"]["heldout_sha256"] for r in reports}) == 3, "reused outer data")
    batch_raw = (directory / f"{PREFIX}batch-native-20261014.json").read_bytes()
    batch = json.loads(batch_raw)
    require(batch["schema"] == "nhsmm-block-mass-smoothing-final-batch-v1", "batch schema")
    require(batch["source_commit"] == SOURCE and batch["seeds"] == list(SEEDS), "batch identity")
    require(
        batch["actual_msteps"] == 171 and batch["oracle_support"] is False, "batch budget/oracle"
    )
    for flag in (
        "matched_initial_pmf_partitions_data_budgets",
        "distinct_seed_training_outer_data",
        "source_software_contract_equivalent",
        "all_numerical_and_solver_checks_pass",
    ):
        require(batch.get(flag) is True, f"batch certificate: {flag}")
    counts = {
        key: sum(r["evaluation"]["arms"][key]["full_gate"] for r in reports)
        for key in ("unsmoothed", "smoothed")
    }
    wins = sum(r["evaluation"]["smoothed_minus_unsmoothed_heldout_ll"] > 0 for r in reports)
    require(
        batch["full_gate_pass_counts"] == counts
        and batch["smoothed_paired_outer_ll_win_count"] == wins,
        "batch counts",
    )
    require([row["seed"] for row in batch["rows"]] == list(SEEDS), "batch row order")
    for report, row in zip(reports, batch["rows"], strict=True):
        seed = str(report["seed"])
        require(batch["inputs"][seed]["sha256"] == hashes[seed], "batch input SHA")
        require(
            row["paired_outer_ll_difference"]
            == report["evaluation"]["smoothed_minus_unsmoothed_heldout_ll"],
            "batch paired LL",
        )
        for key, arm in report["evaluation"]["arms"].items():
            for target, source in (("pmf_tv", "pmf_tv"), ("mean_rmse", "mean_rmse")):
                require(row["arms"][key][target] == arm["metrics"][source], "batch metric mismatch")
            for target, source in (
                ("gain", "heldout_gain_nats_per_row"),
                ("full_gate", "full_gate"),
                ("learned_partition_tv_lower_bound", "learned_partition_tv_lower_bound"),
                ("forward_kl_projection_tv", "forward_kl_projection_tv"),
            ):
                require(row["arms"][key][target] == arm[source], "batch arm mismatch")
    if historical:
        check_archived_bytes(batch_raw, historical_record("smoothing")["batch"])
    return {
        "historical_source_manifest": historical,
        "verified_recorded_evidence": True,
        "independent_numerical_replay_performed_by_this_audit": False,
        "seeds": list(SEEDS),
        "actual_msteps": 171,
        "full_gate_pass_counts": counts,
        "smoothed_outer_ll_wins": wins,
        "input_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/results"))
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Audit recorded source/software identities and sealed bytes; no numerical replay.",
    )
    args = parser.parse_args()
    print(json.dumps(audit_batch(args.results, historical=args.historical), allow_nan=False))


if __name__ == "__main__":
    main()
