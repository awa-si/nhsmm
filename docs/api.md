# Compact model lifecycle

Use NHSMM.fit for model-owned initialization and optimization:

```python
from nhsmm import ModelConfig, NHSMM, save_artifact, load_artifact, HSMMFilterRuntime

model = NHSMM(ModelConfig(
    n_states=3, n_features=5, max_duration=35, causal=True, seed=71,
), device="cpu").fit(sequences)
save_artifact(model, artifact_path)
inference = load_artifact(artifact_path, device="cpu", require_causal=True)
runtime = HSMMFilterRuntime(inference)
posterior = runtime.step(observation, timestamp=timestamp).state_posterior
```

sequences is a dense [B,T,F] or [T,F] tensor, or a list of independent [T,F]
tensors. External context stays explicit through fit(..., context=...). fit
returns the same model, initializes distributions only when absent, then calls
the existing optimize implementation. Existing distribution containers/factories,
configuration checks, restarts, gradients and optimization_summary remain owned
by that implementation. fit does not freeze the model or infer gap/instrument
boundaries. Existing initialize_distributions / optimize calls remain valid.

Do not use predict's default Viterbi decoding as a causal regime feature:
decoding can use later observations within the supplied sequence. Runtime.step
is the causal event surface. Each independent instrument/fold/contiguous stream
needs its own runtime state. Artifact/schema provenance and observation cadence
remain caller responsibilities.

Qlib DatasetH, fit cutoffs, feature hashes, MLflow tracking, folds and Nautilus
events belong to awa-si/nautilus adapter/qlib and adapter/nhsmm.
NHSMMModel now calls the public fit method. Its native TrainerR research command
is python -m adapter.qlib.nhsmm_lab. No Qlib or Nautilus dependency is added here.
NHSMM and GBM share lifecycle verbs, not tensor/target/state semantics or artifact
formats. See the host's docs/model-api.md for the cross-model contract.

## Verification checkpoint — 2026-10-15

On base 251738c28095e590a29b0c24621e3055677e76b3, the full core suite
passed: 1315 tests, 3 CUDA-only skips, zero warnings, 239.83 seconds.
Command: python -m pytest -q tests --tb=short, CPU threads=2.
The focused 8-test fit gate verifies exact parameters, likelihoods, actual
optimization summaries and mode parity against explicit initialization/optimize
for dense/ragged, external/internal context, noncausal and initialized inputs.
Failed fits still propagate errors. No model equation, training budget,
artifact format, oracle setting or frozen holdout was changed.
Log: /work/checkpoints/api-core-full-20261015.log.
