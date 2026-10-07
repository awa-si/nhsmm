# NHSMM Correctness Milestone Handoff

## Repository

- repo: `awa-si/nhsmm`
- branch: `develop`
- active workspace: `7d0ad6575fe8401089ee56be68f32369`
- workspace image: `localhost/workspace:py`
- local git identity: `awa <awa@users.noreply.github.com>`
- remote pushes must use guarded `workspace_repository_push`; never shell `git push`
- Nautilus remains out of scope until this correctness milestone closes

## Milestone status

- Gate 0: COMPLETE — mathematical contract
- Gate 1: COMPLETE — independent exhaustive oracle
- Gate 2: COMPLETE — production likelihood equivalence
- Gate 3: COMPLETE — causal filtering/posterior equivalence
- Gate 4: COMPLETE — Viterbi/MAP equivalence
- Gate 5: COMPLETE — degenerate/structural invariants
- Gate 6: COMPLETE — gradient correctness
- Gate 7: COMPLETE — deterministic property-based state-space exploration
- Gate 8: COMPLETE — numerical precision envelope
- Gate 9: COMPLETE — statistical recovery revalidation
- Gate 10: COMPLETE — final package acceptance

Canonical docs: `docs/hsmm-mathematical-contract.md`, `docs/hsmm-correctness-work-gates.md`, `docs/state.md`, `docs/testing.md`. Independent oracle: `tests/reference/hsmm_exact.py`.

## Key correctness findings and fixes

1. Causal inference is a `(state, age)` dynamic-hazard process; self-transition starts a new episode and resets age to one.
2. Causal prefix likelihood right-censors the final active episode, while non-causal segment likelihood requires a complete final segment ending at `T-1`; these sequence-likelihood contracts are not generally equal.
3. Non-causal duration-dependent Viterbi previously collapsed predecessor duration before destination-specific transition scoring. It now retains `[t,state,duration]` MAP state and maximizes jointly over `(prev_state,prev_duration)`.
4. Non-causal DP temporaries preserve input dtype for Float64 verification.
5. Valid impossible sequence mass remains exact `-inf`, not the Float32 minimum sentinel.

## Gate evidence

Gate 1 adds an independent exhaustive oracle with no production `nhsmm` imports.

Gate 2 checks causal/non-causal likelihood against that oracle across uniform, random, near-deterministic and impossible-support cases, representative `K={1,2,3}`, `D={1,2,3}`, `T<=6`, plus variable-length batches.

Gate 3 checks causal batch forward, `filter_model_sequence`, public `filter_step`, `HSMMFilterRuntime.step`, joint/state/age posteriors, prefix evidence and transition/boundary forecasts against the oracle.

Gate 4 checks causal/non-causal Viterbi against exhaustive MAP, including tied optima, `D=1`, forced support, duration-dependent transitions, self-transition reset and the predecessor-duration regression.

Gate 5 locks structural invariants: `K=1`, `D=1`, `T=1`, exact impossible support, right-padding invariance, batch-vs-single, future-observation invariance, normalization and `age<=D`.

Gate 6 validates Float64 gradients with `torch.autograd.gradcheck` plus central finite differences for initial/duration/transition, duration bias, Gaussian and Student-t parameters. Frozen plain-HSMM controls remain non-trainable.

Gate 7 uses deterministic seed-based property exploration (`seed-N` pytest IDs) because Hypothesis is not a project dependency.

Gate 8 measured precision maxima approximately: Float64 likelihood `1.8e-15`, finite log posterior `1.5e-14`; Float32 likelihood `1.7e-6`, finite log posterior `5.8e-6`; 128-step Float32 batch/filter/runtime log posterior `1.15e-5`. Maintained bounds are `3e-6` likelihood, `1e-5` short log posterior, `2e-5` long log posterior and `<=1e-5` probability error.

Gate 9 reran the canonical statistical harnesses unchanged:

- duration-context: strong PASS, moderate PASS, null-control PASS; median gaps `0.09656`, `0.02716`, `0.01138`
- latent-state recovery: strong PASS (`accuracy 0.99889`, `ARI 0.99661`), moderate PASS (`0.94222`, `0.83349`), null PASS/control (`ARI 0.00287`)
- learning/prediction: strong/moderate/weak usefulness PASS; null learning-control PASS with usefulness intentionally not applicable
- transition-context: strong/moderate/null-control PASS; median correlations `0.999987`, `0.993766`, `0.0`

Gate 10 final package acceptance: Ruff PASS; Black PASS (`86 files would be left unchanged`); `git diff --check` PASS; `python -m pytest -q` PASS with `668 passed, 3 skipped` in the clean Python 3.12 CPU acceptance environment. The skips are CUDA-only.

## Milestone closure

The HSMM correctness milestone is complete through Gate 10. The independent reference suite remains under normal pytest discovery, the final static/full-package gate is green, and the canonical model/testing/state documentation has been reconciled to the observed behavior.

The next work is outside this milestone and must be treated as a separate integration or research gate.

Allowed final claim:

> The maintained HSMM recurrence and outputs are independently verified against exhaustive small-model reference calculations, gradient checks, structural invariants, and measured numerical precision bounds.

Do not claim formal proof, universal statistical identifiability, or downstream trading utility.

## Repository rules

- current repo state is authoritative
- one coherent material work unit per turn
- no architecture refactor during correctness milestone
- math correctness > probabilistic semantics > temporal causality > shape > numerical stability > reproducibility
- full pytest after shared/model-core changes
- Nautilus remains out of scope until milestone closure
- do not revive deprecated `native/`
- PyO3 only where materially justified
- plain HSMM keeps `use_context_encoder=False` by default
