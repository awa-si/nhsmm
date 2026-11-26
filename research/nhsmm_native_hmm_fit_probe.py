"""Research-only full Adam-fit A/B using the restricted native static HMM kernel."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time

import torch
from nhsmm import ModelConfig, NHSMM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-lib", required=True)
    parser.add_argument("--dataset-loader", required=True)
    parser.add_argument("--updates", type=int, default=80)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("btc_reference", args.dataset_loader)
    btc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btc)
    identity, _, digest, z, _, _ = btc.load()
    assert identity == "1158b60ab26a76b5913acc4723051bda536833cda55857ef7f5b2c13b746f53b"
    assert digest == "419bb42fb74d80c526e8eb0f6fd4620f61dfcc311ccbf20a1723142bd4e16b34"
    splits = {"train": z[:335], "validation": z[335:503], "historical_test": z[503:]}
    assert [len(v) for v in splits.values()] == [335, 168, 504]
    x = torch.tensor(splits["train"], dtype=torch.float64)
    config = ModelConfig.for_mode(
        "hmm",
        n_states=3,
        n_features=1,
        seed=7,
        max_iter=args.updates,
        n_init=1,
        verbose=False,
        convergence_stop=False,
        use_scheduler=False,
        optimizer_temperature_mode="constant",
        optimizer_initialization="current",
        loss_bias=0.0,
    )
    torch.manual_seed(7)
    model = NHSMM(config, device="cpu").to(dtype=torch.float64)
    model.initialize_distributions(jitter=0)
    models = (copy.deepcopy(model), copy.deepcopy(model))
    models[1].enable_research_rust(args.rust_lib)

    initial = []
    for m in models:
        m.zero_grad(set_to_none=True)
        ll, loss = m._compute_loss(x, loss_bias=0, it=0, max_iter=args.updates, t_min=1, t_max=1)
        loss.backward()
        initial.append(
            (
                ll.item(),
                {
                    name: p.grad.detach().clone()
                    for name, p in m.named_parameters()
                    if p.grad is not None
                },
            )
        )
    (ll0, g0), (ll1, g1) = initial
    assert abs(ll0 - ll1) < 1e-8 and g0.keys() == g1.keys()
    gradient_max_abs = max((g0[k] - g1[k]).abs().max().item() for k in g0)
    assert gradient_max_abs < 1e-8

    results = []
    for m in models:
        start = time.perf_counter()
        m.optimize(x)
        elapsed = time.perf_counter() - start
        m.eval()
        with torch.no_grad():
            scores = {
                k: float(m.log_likelihood(torch.tensor(v, dtype=torch.float64)).item() / len(v))
                for k, v in splits.items()
            }
        results.append(
            (
                elapsed,
                scores,
                {k: p.detach().clone() for k, p in m.named_parameters()},
                m.optimization_summary(),
            )
        )
    assert all(r[3]["main_updates"] == args.updates for r in results)
    param_max_abs = max(
        (results[0][2][k] - results[1][2][k]).abs().max().item() for k in results[0][2]
    )
    assert param_max_abs < 1e-8
    print(
        json.dumps(
            {
                "dataset_identity": identity,
                "bar_clock_close_sha256": digest,
                "scope": "static finite D=1 CPU Float64, K=3 F=1, one sequence, first-order Adam",
                "updates": args.updates,
                "initial_score_abs": abs(ll0 - ll1),
                "initial_gradient_max_abs": gradient_max_abs,
                "existing": {
                    "fit_seconds": results[0][0],
                    "scores": results[0][1],
                    "optimization": results[0][3],
                },
                "native": {
                    "fit_seconds": results[1][0],
                    "scores": results[1][1],
                    "optimization": results[1][3],
                },
                "final_parameter_max_abs": param_max_abs,
                "fit_time_ratio": results[0][0] / results[1][0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
