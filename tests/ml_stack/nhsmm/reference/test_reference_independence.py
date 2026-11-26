from __future__ import annotations

import ast
from pathlib import Path


def test_exact_reference_has_no_production_nhsmm_imports() -> None:
    path = Path(__file__).with_name("hsmm_exact.py")
    tree = ast.parse(path.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [name for name in imported if name == "nhsmm" or name.startswith("nhsmm.")]
