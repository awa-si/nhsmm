# Stage 2V.5 — Age-tied NHSMM transition context control (2026-10-10)

## Protocol

`trading-regimes-v1`, true episode boundaries, chronological 60/40 split, 415 training and 277 test boundaries, 220 Adam steps (lr 0.012), seeds 2311 and 2312. Compare the actual NHSMM transition distribution with/without one-dimensional external `macro_stress` context. The base transition logits and final context-network projection weights/biases are each tied across all 64 age buckets by symmetric initialization and tied-gradient hooks; actual age-invariance after training was confirmed to be exactly 0 in both arms. Other NHSMM internals are unchanged.

| Age-tied model | Held-out transition MAE | Held-out NLL | Train NLL |
| --- | ---: | ---: | ---: |
| No context, seed 2311/2312 | 0.044460 | 1.055092 | 1.048452 |
| With context, seed 2311 | 0.056146 | 1.090142 | 1.007163 |
| With context, seed 2312 | 0.056369 | 1.091248 | 1.006390 |

The context network lowers training NLL but worsens held-out likelihood and transition-probability MAE even when age-specific parameterization is removed. This supports *context overfitting on this particular dataset and split*, not a general defect in contextual NHSMM transitions. Since the generator itself conditions transitions on macro-stress, a follow-up should test regularization, controlled context-effect strength and non-oracle causal filtering. No production defaults are changed. These supervised boundary results are **not** an `NHSMM.optimize()` or latent-state recovery validation.

Reproduce: `PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_transition_tied_context.py`. Machine-readable output: `docs/validation/results/age-tied-context-2026-10-10.json`.
