"""Five mandatory synthetic NHSMM model-core test groups for AWA workspace.

No Nautilus imports, market data, or GitHub Actions execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path("tests/ml_stack/nhsmm")
GROUPS = ("filtering-a", "filtering-b", "reference", "training", "support")
FILTERING_A = frozenset(
    {
        "test_hsmm_extended_validation.py",
        "test_hsmm_boundary_derivatives.py",
    }
)
SUPPORT_DIRS = frozenset({"artifacts", "context", "contracts", "distributions", "runtime"})


def classify(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if len(relative.parts) != 2:
        raise ValueError(f"unexpected NHSMM model test structure: {relative}")
    category = relative.parts[0]
    if category == "filtering":
        return "filtering-a" if path.name in FILTERING_A else "filtering-b"
    if category in ("training", "reference"):
        return category
    if category in SUPPORT_DIRS:
        return "support"
    raise ValueError(f"unknown NHSMM model test category: {relative}")


def groups() -> dict[str, tuple[str, ...]]:
    files = sorted(ROOT.rglob("test_*.py"))
    if not files:
        raise RuntimeError("NHSMM model tests missing")
    result: dict[str, list[str]] = {key: [] for key in GROUPS}
    for p in files:
        result[classify(p)].append(p.as_posix())
    flattened = [x for names in result.values() for x in names]
    if len(flattened) != len(set(flattened)) or not all(result.values()):
        raise RuntimeError("NHSMM mandatory model groups incomplete or overlapping")
    return {name: tuple(paths) for name, paths in result.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    entries = groups()
    if args.verify:
        for group, paths in entries.items():
            print(f"{group}: {len(paths)} files", file=sys.stderr)
        print(f"TOTAL: {sum(map(len, entries.values()))} core files", file=sys.stderr)
    if args.group:
        print("\n".join(entries[args.group]))
    if not args.group and not args.verify:
        parser.error("specify --group or --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
