# Learning and prediction validation — 2026-10-04

Status: **PASS**

## Purpose

This validation proves that the package does more than execute training and prediction code paths. It checks that optimization changes model state, improves held-out probabilistic fit, recovers latent states on unseen sequences when signal is identifiable, does not invent predictive state identity under a null signal, preserves validation provenance, and respects variable-length prediction semantics.

The benchmark is synthetic and domain-neutral. It does not establish trading utility.

## Harness

Canonical command:

    python scripts/validate_learning_prediction.py --scenario all --seeds 401,402,403,404,405 --max-iter 40 --workers 4 --output /tmp/nhsmm-learning-prediction.json

Configuration:

- K=3, F=3, max_duration=25
- 8 training sequences x 180 timesteps
- 5 strictly disjoint OOS sequences x 180 timesteps
- OOS generation uses seed + 10,000
- n_init=1, max_iter=40, causal model
- emission_init_mode="kmeans" for hard state-identification acceptance
- state accuracy is permutation-invariant

For every run the harness records parameter L1 movement, OOS log likelihood before/after optimize(), OOS accuracy before/after training, OOS ARI, train accuracy, train-to-OOS generalization gap, train/OOS ModelHealthReport, validation fingerprint contracts, and variable-length Viterbi output lengths.

## Results

| Scenario | Separation | Median OOS LL before | Median OOS LL after | Median LL gain | Median accuracy before | Median OOS accuracy | Median accuracy gain | Median OOS ARI | Collapsed OOS runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong | 4.0 | -11.8535 | -4.2253 | +7.8224 | 0.6556 | 0.9978 | +0.3433 | 0.9934 | 0/5 |
| moderate | 2.0 | -6.2799 | -4.1586 | +2.1719 | 0.6233 | 0.9656 | +0.3367 | 0.8979 | 0/5 |
| weak | 1.0 | -5.0041 | -4.1056 | +1.1146 | 0.5333 | 0.7611 | +0.1733 | 0.4085 | 1/5 |
| null | 0.0 | -4.8238 | -3.4228 | +1.0093 | 0.3733 | 0.3589 | -0.0144 | 0.0000 | 5/5 |

All 20 runs changed model parameters.

All 20 runs preserved the validation provenance contract: same fitted-model fingerprint for train and OOS snapshots, but different data fingerprints.

All 20 runs preserved variable-length Viterbi prediction lengths for the [53, 101] probe.

## Interpretation

The signal scenarios demonstrate actual learning rather than simple code-path execution. Optimization changes parameters in every run, held-out likelihood improves materially, held-out state identity improves materially for moderate and weak signals, and strong OOS recovery is effectively saturated after training.

The null scenario is the required negative control. Optimization still improves density fit, but latent-state identity remains at chance-like accuracy with median ARI 0.0. All five null runs collapse to an effectively single-state explanation under the maintained health detector.

Weak separation is intentionally a stress boundary. One of five OOS fits is classified as collapsed while median recovery remains materially above chance. Acceptance allows up to 40% collapse only for this weak scenario. Strong and moderate acceptance allow none.

## Default spread initialization probe

The package default remains emission_init_mode="spread". A supplementary moderate-signal probe over seeds 451..453 showed held-out likelihood improvement in all three runs:

| Seed | Accuracy before | Accuracy after | OOS LL before | OOS LL after | LL gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 451 | 0.9100 | 0.7000 | -6.2844 | -4.5040 | +1.7804 |
| 452 | 0.6189 | 0.3933 | -6.1613 | -4.2551 | +1.9062 |
| 453 | 0.4756 | 0.9600 | -6.5163 | -4.2443 | +2.2720 |

This confirms probabilistic learning under the default initialization, but reconfirms the documented local-minimum / state-identification sensitivity of spread. The hard state-recovery acceptance therefore continues to opt into K-Means initialization rather than silently changing the package default.

## Supported claim

Within this controlled synthetic setting, the package updates learned parameters during optimization, improves held-out sequence likelihood, predicts latent states on unseen sequences when identifiable signal exists, generalizes beyond training sequences, does not show false latent-state recovery under the null control, exposes collapse when data do not support the requested state structure, and preserves variable-length prediction and validation provenance contracts.
