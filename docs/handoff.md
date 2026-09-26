# NHSMM handoff

This file is the short operational handoff for the next work session. `docs/state.md` remains the repository readiness/status owner; this file records only the immediate continuation point and recent performance context.

## Current continuation point

Repository: `awa-si/nhsmm`

Branch: `develop`

Current verified performance work is complete through the incremental encoder overhead cleanup. The next task is **not another encoder micro-optimization**.

Resume with a fresh local profile of the complete steady-state `HSMMFilterRuntime.step()` path and identify the largest remaining end-to-end cost among:

- incremental encoder;
- emission scoring;
- duration/transition boundary scoring;
- normalized filter update;
- runtime state/timestamp/validation handling.

Only optimize the largest measured block when a same-host before/after benchmark shows a material end-to-end improvement.

## Latest verified state

The canonical `DefaultEncoder` streaming path now includes:

- one-window causal convolution evaluated with the existing Conv1d weights as an equivalent linear operation;
- direct one-step LSTM recurrence using the existing `nn.LSTM` parameters;
- inference-only fused input/hidden LSTM gate projection;
- no eval-mode dropout identity calls;
- bounded convolution history retained without an unnecessary clone.

The distribution/runtime path already includes:

- analytical diagonal-Gaussian emission scoring;
- version-keyed cached Gaussian variance/log-normalizer;
- cached duration gate terms;
- no-op ergodic transition mask bypass;
- prepared frozen context-affine fusion;
- internal normalized filter kernel;
- internal normalized filter-state construction without duplicate public validation.

Do not reimplement or duplicate these fast paths.

## Latest direct performance evidence

Same-host local encoder A/B for the latest overhead cleanup:

```text
stream_step()
mean  ~0.06309 ms -> ~0.05209 ms
p50   ~0.06385 ms -> ~0.05192 ms
```

This is approximately a 17% encoder-step reduction. Multi-step output, hidden state, cell state, and convolution history matched the previous path exactly in the exercised local comparisons.

Latest clean-runner integration:

```text
GitHub Actions run 36266084169
62 passed in 2.65s
artifact-loaded runtime:
  p50   0.873408 ms
  p95   0.906474 ms
  mean  0.877417 ms
  Python traced peak 23,536 bytes
  runtime state 151 tensor elements
```

Hosted-runner latency is integration/reference evidence only. Do not compare it causally with a different hosted runner or local machine.

## Rejected / low-value directions

Do not retry these without new profiling evidence:

- in-place LSTM gate activations: isolated activation microbenchmark improved, but real `stream_step()` became slightly slower;
- stacking/fusing all distribution context networks into one larger operation: prior local test was slightly slower;
- naive additional context-linear folding beyond the existing prepared affine fast path: no reliable end-to-end gain;
- replacing tiny `log_softmax` calls merely because they appear individually expensive: require a demonstrated end-to-end win first.

## Performance workflow

1. Materialize current `develop` into `/tmp/nhsmm` using the GitHub workspace/connector path.
2. Prefer local editable install. If network/build isolation blocks installation, an existing dependency-complete local runtime may be used for profiling, but do not call that a clean dependency validation.
3. Establish a fresh steady-state full-runtime baseline before editing.
4. Attribute the baseline to major components rather than optimizing from source inspection alone.
5. Compare before/after on the same host, same model shape, same warmup and measurement loop.
6. Keep latency and `tracemalloc` allocation passes separate.
7. Preserve probabilistic semantics, causal ordering, public APIs, artifact schema, and state-dict compatibility unless a change is explicitly justified.
8. Validate optimized private paths against their public/reference implementations.
9. Run the full discovered pytest suite before completion.
10. Use GitHub Actions only for the final clean-runner integration check, then restore the canonical workflow trigger immediately.

## After the performance pass

Once full-runtime profiling no longer exposes a clear, low-risk material bottleneck, stop micro-optimizing and proceed to the Nautilus-facing evaluation harness described in `docs/state.md`:

```text
A0  no temporal model
B1  Nautilus raw HMM forward filter
B2  HMM + current persistence/duration shaping
C   causal NHSMM
```

Do not map NHSMM latent states to H1 structure classes during the core evaluation.
