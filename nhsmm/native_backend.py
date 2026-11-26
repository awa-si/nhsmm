"""Discover or build packaged native likelihood kernels for the automatic backend."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _kind(mode: str) -> str:
    return "hmm" if mode == "hmm" else "hsmm"


def _suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def native_library_path(mode: str) -> Path | None:
    """Return a usable library, compiling the packaged Rust source if possible.

    The environment override is intended for custom builds and deterministic
    testing. An absent compiler or unwritable cache is an ordinary auto fallback.
    """
    kind = _kind(mode)
    override = os.environ.get(f"NHSMM_NATIVE_{kind.upper()}_LIB")
    if override:
        return Path(override).expanduser()
    source = Path(__file__).parent / "native" / f"{kind}.rs"
    if not source.is_file():
        return None
    rustc = shutil.which("rustc")
    if rustc is None:
        return None
    try:
        version = subprocess.run(
            [rustc, "--version"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
        digest = hashlib.sha256(source.read_bytes() + version.encode()).hexdigest()[:20]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nhsmm" / "native"
        target = cache / f"libnhsmm_{kind}_{digest}{_suffix()}"
        if target.is_file():
            return target
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            candidate = Path(temporary) / target.name
            subprocess.run(
                [
                    rustc,
                    "--edition=2024",
                    "-C",
                    "opt-level=3",
                    "--crate-type=cdylib",
                    str(source),
                    "-o",
                    str(candidate),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            os.replace(candidate, target)
        return target
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=8)
def load_native_kernel(mode: str, path: str):
    if mode == "hmm":
        from nhsmm.models.native_hmm import load_kernel
    else:
        from nhsmm.models.native_hsmm import load_kernel
    return load_kernel(path)
