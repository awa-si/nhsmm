# General-context detect→refine package-core confirmation — 2026-09-27

Package core base: `efd4826ddc48586c47debf075d1c6332596cb0de`  
Frozen reference spec: `632d120c60847ffbe180e5f62e0d1c3820a5332b2b69830b7c5f55a712426389`  
Selection threshold: `-0.07` nats/validation-boundary  
Refinement: `20` transition-context-only steps at lr `0.03`  
Seeds: `731..735`  
CI: **not used**

## Topology-equivalent scoring

The frozen reference generator structurally forbids self-boundary transitions. The canonical package `ergodic` topology permits them. For recovery metrics only, learned package destination probabilities are therefore conditioned on an actual state change (diagonal removed and off-diagonal probabilities renormalized). Selection evidence remains the canonical package likelihood, with no scoring transformation.

| Scenario | selected | Accuracy | ARI | Transition corr | Transition MAE | Context amp | Raw MAE | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strong | 5/5 | 1.000 | 1.000 | 0.919 | 0.092 | 0.706 | 0.132 | PASS |
| moderate | 4/5 | 1.000 | 1.000 | 0.730 | 0.098 | 0.474 | 0.128 | FAIL |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.018 | 0.000 | 0.112 | PASS |

## Frozen-gate result

- Strong: **PASS** — 5/5 selected, 5/5 per-seed recovery passes.
- Moderate: **FAIL** only on median transition correlation: `0.730` vs frozen minimum `0.800`. Selection is 4/5 (minimum 4); 4/5 seeds satisfy the per-seed corr/MAE gates; median MAE `0.098` passes the `0.100` gate.
- Null: **PASS** — 0/5 selected; median context amplitude `0`; conditional transition MAE `0.018`.

**Overall status: PACKAGE-CORE PARTIAL / NOT PASS.** The failure is narrowly localized to moderate-strength 2D transition-function shape fidelity. It is not a state-recovery failure, not a null false-positive failure, and not a selection-count failure. No gate, threshold, or refinement hyperparameter was changed after the frozen reference spec.

## Next diagnostic boundary

Do not retune the gate. Isolate why the package context branch under-recovers moderate functional shape despite selecting 4/5 seeds: compare pre-vs-post refinement functional correlation, context-network parameterization, and duration/self-boundary identifiability. Any package change must improve moderate correlation without reopening strong or null.
