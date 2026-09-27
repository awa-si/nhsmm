# Multi-component component-matched package screen — 2026-09-27

Source package core: `awa-si/nhsmm` `develop` pinned at `efd4826ddc48586c47debf075d1c6332596cb0de`. No CI.

## Harness correction

The isolated A/B/C ablations must remove disabled mechanisms from both the model **and** synthetic ground truth. The prior isolated C screen disabled duration context in the model while still generating duration-context effects (`6 -> 16`), allowing the package to absorb omitted duration structure through valid self-boundary transitions. That produced an invalid absolute transition-MAE failure.

The corrected component-matched generator nulls omitted mechanisms while leaving full-joint D unchanged.

## Corrected package 3-seed screen — seeds 501..503

All strong, moderate, and null A/B/C/D cases pass the frozen screen after component matching. Representative medians:

- strong C: matched accuracy `0.9993`, ARI `0.9979`, transition correlation `0.988`, transition MAE `0.187` — PASS;
- strong D: accuracy/ARI `1.000/1.000`, duration gap `4.599`, transition correlation `0.996`, transition MAE `0.161` — PASS;
- moderate D: accuracy/ARI `1.000/1.000`, duration gap `2.187`, transition correlation `0.938`, transition MAE `0.132` — PASS;
- null D: accuracy/ARI `1.000/1.000`, duration gap `-0.116`, null transition delta `0.0836` — PASS.

The `noncollapsed >= 13/15` production threshold is not applied literally to a three-seed smoke/screen; for `n=3`, all three seeds must be noncollapsed.

## Existing 15-seed evidence

Full-joint D is already package-core PASS on seeds 501..515 for strong, moderate, and null. A completed component-matched strong-C 15-seed aggregate also passes its frozen median gates (`median transition MAE 0.18385`, `median transition corr 0.99602`, `15/15` noncollapsed), although several individual subprocess exit codes are non-zero because single-seed jobs were evaluated against production aggregate thresholds; the aggregate evidence is the relevant artifact.

## Status

**CHECKPOINT:** corrected 3-seed A/B/C/D package screen PASS. Full 15-seed A/B/C confirmation remains the active task; D is frozen and already accepted.
