from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
import time
import tracemalloc
from typing import Optional

import torch

from nhsmm.models import NHSMM
from nhsmm.runtime import HSMMFilterRuntime


@dataclass(frozen=True)
class RuntimeBenchmark:
    steps: int
    warmup_steps: int
    batch_size: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    python_peak_bytes: int
    runtime_state_tensor_elements: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _state_tensor_elements(runtime: HSMMFilterRuntime) -> int:
    state = runtime.state
    if state is None:
        return 0
    total = state.observations.numel()
    total += state.duration_log_prob.numel()
    total += state.transition_log_prob.numel()
    total += state.filter_state.log_posterior.numel()
    encoder_state = state.encoder_state
    if encoder_state is not None:
        total += encoder_state.conv_history.numel()
        total += encoder_state.hidden.numel()
        total += encoder_state.cell.numel()
    return int(total)


def _draw_inputs(
    generator: torch.Generator,
    *,
    steps: int,
    batch_size: int,
    n_features: int,
) -> torch.Tensor:
    return torch.randn(
        steps,
        batch_size,
        n_features,
        generator=generator,
        device="cpu",
    )


def benchmark_cpu_runtime(
    model: NHSMM,
    *,
    steps: int = 512,
    warmup_steps: int = 64,
    batch_size: int = 1,
    seed: Optional[int] = 7,
) -> RuntimeBenchmark:
    """Benchmark incremental CPU latency and Python heap allocation separately.

    Latency is measured without ``tracemalloc`` because Python allocation
    tracing materially perturbs one-step runtime timings. Heap allocation uses
    a separate runtime pass over pre-generated observations.
    """

    if model.device.type != "cpu":
        raise ValueError("CPU runtime benchmark requires a model on CPU")
    if model.training:
        raise RuntimeError("benchmark requires an inference-prepared eval model")
    if steps < 2 or warmup_steps < 1 or batch_size < 1:
        raise ValueError("steps>=2, warmup_steps>=1, and batch_size>=1 are required")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    n_features = int(model.config.n_features)

    warmup_inputs = _draw_inputs(
        generator,
        steps=warmup_steps,
        batch_size=batch_size,
        n_features=n_features,
    )
    latency_inputs = _draw_inputs(
        generator,
        steps=steps,
        batch_size=batch_size,
        n_features=n_features,
    )
    allocation_inputs = _draw_inputs(
        generator,
        steps=steps,
        batch_size=batch_size,
        n_features=n_features,
    )

    runtime = HSMMFilterRuntime(model)
    for x in warmup_inputs:
        runtime.step(x)
    baseline_elements = _state_tensor_elements(runtime)

    latencies_ns: list[int] = []
    for x in latency_inputs:
        start = time.perf_counter_ns()
        runtime.step(x)
        latencies_ns.append(time.perf_counter_ns() - start)

    final_elements = _state_tensor_elements(runtime)
    if final_elements != baseline_elements:
        raise RuntimeError(
            "runtime state size grew with prefix length; incremental bounded-state contract violated"
        )

    allocation_runtime = HSMMFilterRuntime(model)
    for x in warmup_inputs:
        allocation_runtime.step(x)
    tracemalloc.start()
    try:
        for x in allocation_inputs:
            allocation_runtime.step(x)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    allocation_elements = _state_tensor_elements(allocation_runtime)
    if allocation_elements != baseline_elements:
        raise RuntimeError(
            "runtime state size grew during allocation pass; incremental bounded-state contract violated"
        )

    latencies_ms = [value / 1_000_000.0 for value in latencies_ns]
    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return RuntimeBenchmark(
        steps=steps,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        p50_ms=float(statistics.median(latencies_ms)),
        p95_ms=float(ordered[p95_index]),
        mean_ms=float(statistics.fmean(latencies_ms)),
        python_peak_bytes=int(peak),
        runtime_state_tensor_elements=final_elements,
    )
