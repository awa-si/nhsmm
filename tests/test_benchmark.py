from __future__ import annotations

from nhsmm import ModelConfig, NHSMM, prepare_inference
from nhsmm.benchmark import benchmark_cpu_runtime


def test_cpu_benchmark_reports_bounded_incremental_state() -> None:
    config = ModelConfig(
        n_states=2,
        n_features=3,
        max_duration=4,
        causal=True,
        dropout=0.0,
        seed=47,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    prepare_inference(model)

    result = benchmark_cpu_runtime(
        model,
        warmup_steps=3,
        steps=8,
        batch_size=1,
        seed=53,
    )

    assert result.steps == 8
    assert result.p50_ms > 0.0
    assert result.p95_ms > 0.0
    assert result.python_peak_bytes >= 0
    assert result.runtime_state_tensor_elements > 0
