# Component-matched multi-component package-core confirmation — 2026-09-27

Source package core: SHA-pinned `awa-si/nhsmm` develop sources; local-only, no CI.

The A/B/C generators were corrected so mechanisms disabled in the model are also nulled in the synthetic truth. D remains the already accepted full-joint generator. Frozen seeds: `501..515`; metric thresholds unchanged.

| Scenario | Ablation | Acc | ARI | Duration gap | Transition corr | Transition MAE | Null transition Δ | Noncollapsed | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong | A | 1.000 | 1.000 | 0.000 | 0.000 | 0.189 | 0.000 | 15/15 | PASS |
| strong | B | 1.000 | 1.000 | 4.426 | 0.000 | 0.163 | 0.015 | 15/15 | PASS |
| strong | C | 1.000 | 1.000 | 0.000 | 0.996 | 0.184 | 0.270 | 15/15 | PASS |
| moderate | A | 1.000 | 1.000 | 0.000 | 0.000 | 0.189 | 0.000 | 15/15 | PASS |
| moderate | B | 1.000 | 1.000 | 2.709 | 0.000 | 0.168 | 0.010 | 15/15 | PASS |
| moderate | C | 1.000 | 1.000 | 0.000 | 0.963 | 0.162 | 0.136 | 15/15 | PASS |
| null | A | 1.000 | 1.000 | 0.000 | 0.000 | 0.189 | 0.000 | 15/15 | PASS |
| null | B | 1.000 | 1.000 | -0.125 | 0.000 | 0.176 | 0.005 | 15/15 | PASS |
| null | C | 1.000 | 1.000 | 0.000 | 0.000 | 0.155 | 0.044 | 13/15 | PASS |

All nine component-matched A/B/C gates PASS.

Previously accepted full-joint D also passes strong/moderate/null across all 15 frozen seeds. Therefore multi-component identifiability is accepted at package-core level across isolated and joint mechanisms.

Important diagnostic closure: old Strong-C MAE `0.2303` was caused by leaving duration-context truth active while disabling duration context in the model. Under component-matched C, Strong transition MAE is `0.1838`, corr `0.996`, direction `1.0`, with 15/15 noncollapsed.

Package-core scope remains the exact external-context training/likelihood/distribution/filtering dependency closure; full encoder/import-stack certification is outside this gate.
