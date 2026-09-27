# Package validation: context-conditioned duration recovery

## Scope

This document records package-level empirical evidence for NHSMM training and causal duration forecasting. It is independent of Nautilus, market data, trading targets, and downstream supervised models.

## Controlled synthetic contract

Synthetic sequences use three observed coordinates and two latent states. One directly observable context coordinate switches between two regimes. The generator changes the episode-duration distribution by context while state emissions remain separately identifiable. The package is therefore tested against known duration ground truth rather than an inferred external label.

Training uses `NHSMM.optimize()` with a causal `DefaultEncoder`, `K=2`, `D=30`, learning rate `1e-2`, one initialization per seed, and a 40-iteration maximum budget. Evaluation is performed on separately generated sequences.

The primary diagnostic is

`mean P(end within 5 | short-duration context) - mean P(end within 5 | long-duration context)`.

A positive value is the expected direction. State occupancy is also checked to reject a trivial one-state collapse.

## Local package-validation evidence

The validation sequence was executed locally from package code with no Nautilus dependency.

### Optimizer coverage

The causal encoder participates in the likelihood graph and must be part of the optimizer. A regression check snapshots all trainable encoder parameters, runs one optimization step, and verifies that encoder parameters are present in the optimizer and at least one changes.

### Training-budget ablation

A nine-seed controlled ablation compared the earlier short training budget with longer training and several duration-context modifications. The unmodified model with `max_iter=40` was selected; no auxiliary duration loss, gradient multiplier, or alternative duration parameterization was required.

For the selected 40-iteration baseline over seeds 101-109:

- positive short-vs-long hazard direction: 9/9;
- material hazard gap (`>= 0.02`): 9/9;
- correct duration-PMF direction: 9/9;
- non-collapsed latent-state usage: 9/9;
- median horizon-5 hazard gap: approximately `0.118`;
- mean validation log likelihood per row: approximately `-2.877`, versus approximately `-3.560` for the 18-iteration baseline.

This identifies insufficient optimization budget, rather than a demonstrated need for a new objective, as the cause of the earlier weak recovery on this controlled task.

### Independent multi-seed acceptance

The selected configuration was evaluated on new seeds 201-215 under three separately generated scenarios:

| Scenario | Ground-truth means | Positive gap | Gap >= 0.02 | Median gap | Non-collapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| strong context | 6 vs 18 | 14/15 | 14/15 | 0.0923 | 15/15 |
| moderate context | 9 vs 15 | 13/15 | 10/15 | 0.0380 | 15/15 |
| null context | 12 vs 12 | descriptive only | 4/15 | 0.00883 | 15/15 |

The null-context acceptance guard required both absolute median and absolute mean hazard gaps to remain at or below `0.015`; it passed. This guards against a training procedure that manufactures separation when the generator contains no duration-context effect.

### Runtime invariants

The selected package path also passed 200/200 randomized survival/transition invariant checks, including probability normalization, horizon monotonicity, one-step episode-end consistency, and causal prefix invariance.

## Interpretation

The evidence supports a narrow package claim: under controlled synthetic ground truth, the current causal NHSMM can learn context-conditioned episode-duration structure and recover the expected survival/hazard ordering robustly across independent seeds, without latent-state collapse or spurious null-context separation.

It does **not** establish that an NHSMM will provide useful latent states or hazard forecasts for a particular downstream domain. Domain usefulness remains an external out-of-sample evaluation question.

## Reproduction policy

The maintained unit suite checks optimizer/encoder coverage and the training-budget default. Full multi-seed empirical acceptance is intentionally kept out of routine pytest because it is an expensive research validation. Re-run the controlled acceptance whenever the training objective, duration parameterization, context encoder, causal filter/survival semantics, or optimizer coverage changes materially.

Use:

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
```
