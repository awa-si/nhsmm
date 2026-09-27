# General-context split-fit direction replication confirmatory — 2026-09-27

Status: **PACKAGE-CORE PASS**.

This candidate addresses the remaining general-context detection defect by requiring an effect to reproduce across two independent fits trained on disjoint halves of the training data. Detection is truth-free: states are aligned by learned emission centers, then the context-induced transition-delta function is compared on a fixed context grid.

## Frozen detector

Development seeds: `841..845`.

Frozen once on the development block:

- replication correlation `>= 0.55`;
- minimum replicated functional amplitude across the two half-fits `>= 0.50`.

Fresh confirmatory seeds: `851..855`.

No threshold change was made after the confirmatory block started.

## Confirmatory result

| Scenario | Selected | Median replication corr | Median min-half amplitude | Median transition corr | Median transition MAE | Accuracy | ARI | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 0.933 | 0.800 | 0.976 | 0.051 | 1.000 | 1.000 | PASS |
| moderate | 4/5 | 0.771 | 0.516 | 0.914 | 0.071 | 1.000 | 1.000 | PASS |
| null | 0/5 | 0.426 | 0.363 | 0.000 | 0.028 | 1.000 | 1.000 | PASS |

Moderate seed `851` was correctly rejected by the frozen detector (`replication_corr ~0`, `min-half amplitude 0.440`), leaving `4/5` selected as required. The four selected moderate seeds recover the transition-context function with strong shape fidelity and low MAE. Null remains `0/5` selected.

## Interpretation

The remaining blocker was not estimator expressiveness but reproducibility of evidence. A single fit can produce structured context modulation under null, but those effects do not replicate with both sufficient directional agreement and sufficient amplitude across independent split fits. Strong and moderate true effects do.

This closes the controlled general-context package-core gate for the tested 2D external-context setting. The result does not establish downstream trading value or domain semantics.
