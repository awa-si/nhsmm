# General-context robustness deep dive — 2026-09-27

Scope: **local independent causal-HSMM reference evidence**, not package-level acceptance. No CI was used.

## Frozen confirmatory design

- Context families: continuous scalar and causal 2D.
- `K=3`, `D=24`, `T=120`, 24 training sequences, 6 evaluation sequences.
- Confirmatory seeds: `611..615`.
- Pre-run spec SHA256: `aac3a0436f7af87b738f188ac40bb124276070479da205c378ea66ff139b66e0`.
- Active gate: median accuracy >=0.95, ARI >=0.90, transition correlation >=0.80, transition MAE <=0.10; >=4/5 seed-level passes.
- Null gate: median context amplitude <=0.15, transition MAE <=0.08; >=4/5 seed-level passes.

## Unregularized confirmatory results

| Scenario | Accuracy | ARI | Transition corr | Transition MAE | Context amplitude | Seed passes | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| scalar strong | 1.000 | 1.000 | 0.972 | 0.063 | 0.512 | 5/5 | PASS |
| scalar moderate | 1.000 | 1.000 | 0.955 | 0.041 | 0.314 | 4/5 | PASS |
| scalar null | 1.000 | 1.000 | n/a | 0.043 | 0.122 | 5/5 | PASS |
| 2D strong | 1.000 | 1.000 | 0.954 | 0.086 | 0.657 | 5/5 | PASS |
| 2D moderate | 1.000 | 1.000 | 0.833 | 0.077 | 0.391 | 4/5 | PASS |
| 2D null | 1.000 | 1.000 | n/a | 0.067 | 0.237 | 3/5 | **FAIL** |

Overall unregularized gate: **FAIL (5/6 scenarios pass)**. The failure is isolated to spurious context response under 2D null truth; state recovery remains intact.

## Capacity / sample-efficiency finding

A lower-data discovery screen showed substantially larger null amplitudes. Increasing controlled training sequences reduced but did not eliminate the 2D null effect. This points to boundary-sample variance / context capacity rather than state-identifiability failure.

The maintained package binary transition-context validator uses `max_duration=1`; here `D=24`. A duration-dependent transition context therefore has many more context-conditioned degrees of freedom while the effective evidence is concentrated at episode boundaries.

## L2 rescue experiment — rejected

Development seeds `621..623` scanned L2 penalties on transition-context weights. `lambda=0.1` was selected because it was the first tested value to push development null amplitude below `0.15` while retaining development moderate correlation >`0.80` and MAE <`0.10`.

The choice was frozen before confirmatory seeds `631..635`; confirmatory spec SHA256: `d208aec17deb752d2c373c645f18529d7a66c0a16b8a1389e21d744daa5d1f8e`.

| Scenario | Accuracy | ARI | Transition corr | Transition MAE | Context amplitude | Seed passes | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 2D strong | 1.000 | 1.000 | 0.915 | 0.153 | 0.467 | 1/5 | **FAIL** |
| 2D moderate | 1.000 | 1.000 | 0.846 | 0.104 | 0.271 | 5/5 | **FAIL** |
| 2D null | 1.000 | 1.000 | n/a | 0.052 | 0.140 | 5/5 | PASS |

Simple global L2 shrinkage solves the null gate but over-shrinks real context effects. It is **not accepted** as a package change.

## Current research conclusion

- Continuous scalar general-context robustness: **reference PASS**.
- 2D active-context recovery: **reference PASS**.
- 2D null suppression: **reference FAIL**.
- Global L2 rescue: **REJECTED**.
- Do not relax gates or promote this slice yet.

Next: isolate context capacity versus boundary count and evaluate a mechanism that can suppress unsupported multidimensional context without uniformly shrinking real effects (for example explicit context selection/gating or a lower-rank/shared context parameterization), then confirm on fresh seeds before touching package defaults.
