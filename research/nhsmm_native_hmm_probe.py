"""Research-only PyTorch versus opt-in Rust HMM likelihood and gradient probe."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import time

import torch
from nhsmm import ModelConfig, NHSMM


def measure(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return {
        "repeats": repeats,
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (repeats - 1))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-lib", required=True)
    parser.add_argument("--dataset-loader", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("btc_reference", args.dataset_loader)
    btc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btc)
    identity, _, digest, z, _, _ = btc.load()
    assert identity == "1158b60ab26a76b5913acc4723051bda536833cda55857ef7f5b2c13b746f53b"
    assert digest == "419bb42fb74d80c526e8eb0f6fd4620f61dfcc311ccbf20a1723142bd4e16b34"
    x = torch.tensor(z[:335], dtype=torch.float64)
    config = ModelConfig.for_mode("hmm", n_states=3, n_features=1, seed=7)
    model = NHSMM(config, device="cpu").to(dtype=torch.float64)
    model.initialize_distributions(jitter=0)
    base, native = copy.deepcopy(model), copy.deepcopy(model)
    base.set_backend("python")
    native.set_backend("rust", library_path=args.rust_lib)

    def graph(m):
        m.zero_grad(set_to_none=True)
        value = m.log_likelihood(x, reduce=True)
        value.backward()
        return value.item(), {
            k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None
        }

    (score0, grads0), (score1, grads1) = graph(base), graph(native)
    score_abs = abs(score0 - score1)
    assert grads0.keys() == grads1.keys()
    grad_abs = max((grads0[k] - grads1[k]).abs().max().item() for k in grads0)
    assert score_abs < 1e-8 and grad_abs < 1e-8
    results = {
        "dataset_identity": identity,
        "bar_clock_close_sha256": digest,
        "scope": "one static finite CPU Float64 HMM sequence, first-order gradient",
        "observations": len(x),
        "states": 3,
        "features": 1,
        "score_abs": score_abs,
        "gradient_max_abs": grad_abs,
        "torch": measure(lambda: graph(base), 12),
        "rust_optin": measure(lambda: graph(native), 40),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
