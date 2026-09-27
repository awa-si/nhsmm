# Multi-component package-core confirmation — 2026-09-27

Source: `awa-si/nhsmm` `develop` @ `efd4826ddc48586c47debf075d1c6332596cb0de`. No CI.

This run materialized the NHSMM core paths used by the external-context validator: training, causal likelihood recursion, duration/transition/emission distributions, and causal filtering. Import/encoder glue was inert because every validation path supplies explicit external context.

## Full-joint D — 15 seeds (501..515)

| Scenario | Acc | ARI | Duration gap | Transition corr | Transition MAE | Null transition Δ | Noncollapsed | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong | 1.000 | 1.000 | 4.383 | 0.997 | 0.169 | 0.306 | 15/15 | PASS |
| moderate | 1.000 | 1.000 | 2.273 | 0.938 | 0.132 | 0.153 | 15/15 | PASS |
| null | 1.000 | 1.000 | -0.116 | 0.000 | 0.130 | 0.040 | 13/15 | PASS |

**Full-joint package-core status: PASS.**

## Ablation diagnostic

Strong C (transition context on, duration context off) retained perfect state recovery and transition direction (`corr=0.994`, direction `1.0`) but absolute transition MAE was `0.230 > 0.20`. Learned boundary transitions carried median ~30.7% self-transition mass. Because the generator still contains context-conditioned duration effects while C removes that mechanism from the model, the package can legally represent omitted duration structure through self-boundary transitions. The C absolute-MAE failure is therefore treated as an ablation-design misspecification, not evidence against full-joint identifiability.

Next: make A/B/C generator semantics component-matched (null omitted mechanisms), keep D unchanged, and rerun the isolated diagnostics without changing frozen metric thresholds.
