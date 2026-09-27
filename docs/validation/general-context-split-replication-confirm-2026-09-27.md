# General-context split-fit direction replication confirmatory — 2026-09-27

Status: **PACKAGE-CORE PASS**.

This candidate addresses the remaining general-context detector instability by requiring reproducibility across two independent fits on disjoint training halves. Detection is truth-free: state alignment uses learned emission centers only, then the context-induced transition-delta functions are compared on a fixed context grid.

## Frozen detector

Chosen only on development seeds `841..845`:

- `replication_corr >= 0.55`
- `min(amplitude_half1, amplitude_half2) >= 0.50`

Fresh confirmatory seeds: `851..855`.

No threshold or model change was made after the confirmatory run started.

## Confirmatory result

| Scenario | Selected | Accuracy | ARI | Median transition corr | Median transition MAE | Median replication corr | Median min half-amplitude | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 1.000 | 1.000 | 0.976 | 0.051 | 0.933 | 0.800 | PASS |
| moderate | 4/5 | 1.000 | 1.000 | 0.914 | 0.071 | 0.771 | 0.516 | PASS |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.028 | 0.426 | 0.363 | PASS |

The complete frozen gate is satisfied: strong detection is preserved, moderate selection reaches the required `>=4/5`, null remains `0/5`, and selected active cases retain high transition-shape fidelity with perfect state recovery.

## Interpretation

The open detector-stability boundary is resolved by requiring replicated context-effect direction and minimum effect size across independent data splits. This is stronger than a single-fit amplitude floor because both direction and magnitude must recur independently.

The earlier rejected candidates remain rejected and were not retuned: effect-floor, fold-sign, single-permutation, permutation-rank/binomial, and Fisher-permutation detectors.

**Conclusion: General-context robustness is PACKAGE-CORE PASS.** The next boundary is API/design translation of the detector/estimator workflow; no further detector tuning is justified by the frozen synthetic gate.
