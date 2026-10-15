import json
from pathlib import Path

import pytest

from scripts.verify_optimizer_bridge import audit, verify

REPORT = Path("docs/validation/results/optimizer-bridge-pilot-native-20261014.json")


@pytest.mark.parametrize("field,value", [("complete", False), ("schema", "wrong")])
def test_report_header_tampering_rejected(tmp_path, field, value):
    report = json.loads(REPORT.read_text())
    report[field] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify(path, historical=True)


def test_changed_source_manifest_rejected(tmp_path):
    report = json.loads(REPORT.read_text())
    report["execution"]["source_sha256"]["nhsmm/models/base.py"] = "wrong"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source manifest"):
        audit(path)


def test_rebound_initial_objective_rejected(tmp_path):
    report = json.loads(REPORT.read_text())
    report["rows"][0]["arms"]["constant"]["history"][0]["target_ll"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="uniform start"):
        audit(path)


def test_registered_historical_audit_never_invokes_replay(monkeypatch):
    import scripts.verify_optimizer_bridge as module

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata audit invoked numerical replay")

    for name in ("aggregate", "sequences", "public_check", "posterior_gradient_check"):
        monkeypatch.setattr(module, name, forbidden)
    original = REPORT.read_bytes()
    result = audit(REPORT)
    assert result["verified_historical_record"] is True
    assert result["independent_terminal_outer_public_replay"] is False
    assert result["verification_optimizer_updates"] == 0
    assert REPORT.read_bytes() == original


def test_numerical_replay_still_rejects_software_drift(monkeypatch):
    import scripts.verify_optimizer_bridge as module

    monkeypatch.setattr(module.platform, "python_version", lambda: "different-interpreter")
    with pytest.raises(ValueError, match="software"):
        verify(REPORT, historical=True)


def test_historical_audit_rejects_changed_recorded_software(tmp_path):
    report = json.loads(REPORT.read_text())
    report["execution"]["python"] = "different-interpreter"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recorded software"):
        audit(path)
