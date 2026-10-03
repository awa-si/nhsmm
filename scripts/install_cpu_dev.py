"""Install the NHSMM development environment with CPU-only PyTorch.

Run this from an activated Python 3.12+ virtual environment:

    python scripts/install_cpu_dev.py

PyTorch selects platform-specific accelerator dependencies on its normal package
index. Installing the CPU wheel first makes the subsequent editable development
install reuse that satisfied dependency instead of resolving CUDA packages.
"""

from __future__ import annotations

import subprocess
import sys

_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def main() -> None:
    _pip("install", "--index-url", _CPU_INDEX, "torch>=2.2")
    _pip("install", "-e", ".[dev]")


if __name__ == "__main__":
    main()
