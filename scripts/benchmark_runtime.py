from __future__ import annotations

import argparse
import json

from nhsmm.artifact import load_artifact
from nhsmm.benchmark import benchmark_cpu_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NHSMM production CPU runtime")
    parser.add_argument("artifact", help="Path to a versioned NHSMM artifact")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    model = load_artifact(args.artifact, device="cpu", require_causal=True)
    result = benchmark_cpu_runtime(
        model,
        steps=args.steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
    )
    print(json.dumps(result.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
