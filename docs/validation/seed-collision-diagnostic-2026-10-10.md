# Stage 2S follow-up — Seed collision diagnostic

## Reproduction

`PYTHONPATH=/workspace OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /work/env/nhsmm/bin/python scripts/diagnose_trading_seed_collision.py --seeds 904 905 906 --output /data/workspace/stage2/seed-collision.json`

Exit code 0, 13.362 seconds. Results: `results/seed-collision-904-906.json`.

## Findings

- K-Means centers have distinct SHA-256 digests for seeds 904, 905, 906.
- Each model has 16 named parameters; 11 have identical initialization hashes across all three seeds, consistent with deterministic constant initialization. The remaining five differ between seeds.
- Consequently, seeds 905 and 906 are **not initialized identically**. Their identical Stage 2S aggregate metrics cannot be explained by a full initialization collision.
- This diagnostic captures the pre-training initialization and a separate K-Means sample, not the final fitted parameter tensors or per-bar state posteriors. Identical aggregate metrics do **not** prove identical fitted models. The collision remains unresolved.

## Next gate

Instrument training to persist final fitted-parameter hashes and per-bar prediction hashes for seeds 905 and 906; compare equality up to state permutation and distinguish label switching from genuinely identical decoded paths. Treat the two seeds as potentially dependent evidence until then.
