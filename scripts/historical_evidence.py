"""Read-only identities for historical evidence, independent of replay software."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = Path("docs/validation/historical-evidence-index.json")


def historical_record(name):
    index = json.loads((ROOT / INDEX).read_text())
    if index.get("schema") != "nhsmm-historical-evidence-index-v1":
        raise ValueError("historical evidence index schema")
    return index["records"][name]


def check_archived_bytes(raw, reference, *, proof=None):
    """Check original report/proof bytes; do not perform a numerical replay."""
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("historical raw SHA mismatch")
    if "proof_path" not in reference:
        return None
    proof_raw = (ROOT / reference["proof_path"]).read_bytes()
    if hashlib.sha256(proof_raw).hexdigest() != reference["proof_sha256"]:
        raise ValueError("historical proof SHA mismatch")
    original = json.loads(proof_raw)
    if proof is not None and proof != original:
        raise ValueError("historical proof content mismatch")
    return original
