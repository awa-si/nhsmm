# Stage 2S follow-up — Fitted seed identity (2026-10-10)

## Reproduction

`PYTHONPATH=/workspace OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /work/env/nhsmm/bin/python scripts/diagnose_trading_fitted_seed_identity.py --seeds 905 906 --max-iter 30 --output /data/workspace/stage2/fitted-seed-identity-905-906.json`

Exit code 0; 1265.68 seconds. Full parameter hashes and comparison metrics: `results/fitted-seed-identity-905-906.json`.

## Findings

- Seeds 905 and 906 agree on 65.8854% of decoded test-window labels without relabeling.
- Optimal state-label permutation `[0,1,3,2]` produces **100% decoded path agreement**; ARI is **1.0**.
- Only 10 of 16 final named-parameter hashes match exactly. Maximum absolute difference between raw, unaligned posterior arrays is 1.0000004768. Thus this establishes equivalence of **argmax decoded paths up to label permutation**, not equivalence of full parameterized models or calibrated posterior probabilities.
- Initial K-Means centers and some initialized parameters differed between seeds (see `seed-collision-diagnostic-2026-10-10.md`).

## Implications

The Stage 2S seed-905/906 metric collision is accounted for by identical decoded paths up to label switching. These two seeds do **not** provide independent evidence of improved state segmentation, despite distinct initializations. Do not count them as independent successes. Next validation should test different datasets/time periods and stronger initialization diversity, and inspect posterior equivalence after state alignment if relevant. No production configuration change is justified by this diagnostic.
