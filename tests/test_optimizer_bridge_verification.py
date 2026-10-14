import json
from pathlib import Path

import pytest

from scripts.verify_optimizer_bridge import verify

REPORT = Path("docs/validation/results/optimizer-bridge-pilot-native-20261014.json")


@pytest.mark.parametrize("field,value", [("complete", False), ("schema", "wrong")])
def test_report_header_tampering_rejected(tmp_path, field, value):
    report = json.loads(REPORT.read_text())
    report[field] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify(path)


def test_changed_source_manifest_rejected(tmp_path):
    report = json.loads(REPORT.read_text())
    report["execution"]["source_sha256"]["nhsmm/models/base.py"] = "wrong"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source manifest"):
        verify(path)


def test_rebound_initial_objective_rejected(tmp_path):
    report = json.loads(REPORT.read_text())
    report["rows"][0]["arms"]["constant"]["history"][0]["target_ll"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="uniform start"):
        verify(path)
