import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_block_mass_smoothing import audit_batch, audit_seed

RESULTS = Path("docs/validation/results")


@pytest.fixture
def evidence():
    stem = RESULTS / "block-mass-smoothing-seed1912"
    report = json.loads(Path(f"{stem}-native-20261014.json").read_text())
    proof = json.loads(Path(f"{stem}-verification-native-20261014.json").read_text())
    return report, proof


def encode(report, proof):
    raw = json.dumps(report, allow_nan=True).encode()
    proof["input_report_sha256"] = hashlib.sha256(raw).hexdigest()
    proof["evaluation"] = copy.deepcopy(report["evaluation"])
    return raw


def test_completed_batch():
    result = audit_batch(RESULTS, historical=True)
    assert result["full_gate_pass_counts"] == {"unsmoothed": 3, "smoothed": 3}
    assert result["smoothed_outer_ll_wins"] == 3
    assert result["independent_numerical_replay_performed_by_this_audit"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["execution"].update(source_commit="wrong"),
        lambda r: r.update(active={"key": "smoothed"}),
        lambda r: r["phases"]["smoothed"].update(strength=0),
        lambda r: r["phases"]["warm"].update(training_sha256="wrong"),
        lambda r: r["phases"]["smoothed"]["history"][1]["solver"].update(
            maximum_normalized_anchored_gradient=1
        ),
        lambda r: r["evaluation"].update(labels_used_in_learning_or_selection=True),
        lambda r: r["evaluation"].update(actual_msteps=58),
        lambda r: r["evaluation"]["arms"]["smoothed"].update(full_gate=False),
        lambda r: r["evaluation"]["arms"]["smoothed"].update(
            heldout_gain_nats_per_row=float("nan")
        ),
        lambda r: r["evaluation"]["arms"]["smoothed"]["public_recursion"].update(ll_error=1),
        lambda r: r["evaluation"].update(smoothed_minus_unsmoothed_heldout_ll=123),
    ],
)
def test_semantic_tampering_even_with_rebound_sha(evidence, mutation):
    report, proof = evidence
    mutation(report)
    raw = encode(report, proof)
    with pytest.raises(ValueError):
        audit_seed(raw, proof, 1912, historical=True)


def test_raw_bytes_tampering(evidence):
    report, proof = evidence
    raw = encode(report, proof)
    with pytest.raises(ValueError, match="SHA"):
        audit_seed(raw + b"\n", proof, 1912, historical=True)


def test_missing_replay_certificate(evidence):
    report, proof = evidence
    raw = encode(report, proof)
    proof["completed_resume_noop"] = False
    with pytest.raises(ValueError, match="replay certificate"):
        audit_seed(raw, proof, 1912, historical=True)


def test_resealed_unregistered_report_change_is_not_historical_evidence(evidence):
    report, proof = evidence
    report["unregistered_note"] = "a new document is not the original evidence"
    raw = encode(report, proof)
    with pytest.raises(ValueError, match="historical raw SHA"):
        audit_seed(raw, proof, 1912, historical=True)


def test_sealed_original_proof_bytes_are_checked(tmp_path, monkeypatch):
    import scripts.historical_evidence as module

    reference = module.historical_record("smoothing")["reports"]["1912"]
    raw = Path(reference["path"]).read_bytes()
    proof_raw = Path(reference["proof_path"]).read_bytes()
    copied = tmp_path / reference["proof_path"]
    copied.parent.mkdir(parents=True)
    copied.write_bytes(proof_raw + b"changed")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="historical proof SHA"):
        module.check_archived_bytes(raw, reference)
