# Stage 2V.4 — Native transition age-tying control (2026-10-10)

## Protocol

Ground-truth-supervised transition-boundary diagnostic on `trading-regimes-v1`; chronological 60/40 split (415 train and 277 test boundaries). Two seeds (2311, 2312), 220 Adam steps at learning rate 0.012, **no context**, `max_duration=64`. Both arms call the actual NHSMM `Transition.log_matrix` distribution. The age-tied arm shares gradient updates across all age-indexed logits, starting from identical age rows; the age-specific arm trains all entries independently. Parameter sharing was verified numerically (`max_age_probability_difference=0`). This is not `NHSMM.optimize()` and does not evaluate the latent-state training objective.

## Results

| Transition parameterization | Test probability MAE | Test NLL | Train NLL |
| --- | ---: | ---: | ---: |
| Empirical pooled row-frequency reference | 0.03771 | 1.02751 | — |
| NHSMM age-tied, seeds 2311 and 2312 | 0.04446 | 1.05509 | 1.04845 |
| NHSMM age-specific, seeds 2311 and 2312 | 0.12605 | 1.43622 | 0.79074 |

Both NHSMM arms started from train NLL 1.38629. The age-specific arm achieves substantially lower *training* loss but worse test likelihood and probability calibration, indicating overfitting in this boundary-supervised setup. The age-tied arm closes most of the gap to the pooled empirical baseline. Both seeds yield identical results because the no-context logits initialize identically and training is deterministic.

## Conclusion and limitations

Age-specific transition parameters are a material driver of the observed generalization deficit in the synthetic dataset. This is evidence of excess parameter flexibility relative to only 415 training boundaries, not evidence of a recurrence/gradient implementation bug. Generator transitions depend on macro-stress, so an age-tied *context-dependent* comparison remains possible. This result is **not** sufficient to change package defaults; validation should also cover controlled data with genuinely age-dependent transitions and integrated unsupervised NHSMM behavior. No source model code or defaults were changed.

## Reproduction

`PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_transition_age_tying.py`

Full JSON: `docs/validation/results/native-age-tied-transition-2026-10-10.json`.
