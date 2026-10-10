# Stage 2V.6 — Age-tied contextual transition regularization

Ground-truth episode-boundary supervised experiment on `trading-regimes-v1` (415 training and 277 held-out boundaries; chronological 60/40 split). Both the base NHSMM transition logits and the output projection of the context network are exactly age-tied. All variants use native `Transition.log_matrix`, Adam with lr=0.012 for 220 updates, and two contextual seeds (2311 and 2312). Weight decay applies to context-network parameters only; no-context baseline has no context network.

| Configuration | Test probability MAE | Test NLL |
| --- | ---: | ---: |
| Age-tied, no context | 0.044460 | 1.055092 |
| Context, weight decay 0 | 0.056146–0.056369 | 1.090142–1.091248 |
| Context, weight decay 0.01 | 0.056762–0.056958 | 1.084809–1.085642 |
| Context, weight decay 0.1 | 0.044001–0.044008 | 1.053404–1.053405 |

Age-invariance was checked after fitting and was exact in every run. Strong context-network weight decay closes the held-out generalization gap in this small synthetic test. The small apparent edge over the no-context arm is not a robust claim of genuine predictive context benefit: a single split and two seeds cannot establish that, and regularization can effectively suppress the context response. This is supervised boundary fitting, **not** `NHSMM.optimize()` or latent-state recovery. No production defaults or model code were changed.

Reproduce: `PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_context_regularization.py`. JSON results: `docs/validation/results/age-tied-context-regularization-2026-10-10.json`.
