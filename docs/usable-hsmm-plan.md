---
title: Usable supervised HSMM
description: Static CPU qualification and subsequent observation-robustness limits.
type: report
status: active
language: en
owner: AWA
created: 2026-10-24
updated: 2026-10-27
---

# Usable supervised HSMM

Status: the declared supervised static HMM/HSMM CPU qualification is complete.
Nine synthetic profiles pass; trading production qualification remains open.
Model source: 8c0998c77d34334a76200ebabd508b6372bc917c. No new market campaign.

1. Boundary-aware training cohorts: require provenance; trim unknown first episode.
2. Censored Beta-smoothed duration hazards and pooled Dirichlet-smoothed transitions.
3. Independent chronological synthetic calibration versus geometric HMM, with fixed train/validation/test splits.
4. Exact second derivatives of supervised filter score at h_tail=1.
5. Full canonical HMM/HSMM gate, package/default import and native serving verification.

Live inference consumes historical observations without validation-label trimming. An explicit stationary state/age prior is available for time-homogeneous unknown-age entry; reset preserves the caller-selected entry policy.
Existing ordinary APIs keep their fresh-episode contract. No arbitrary initial-age law is guessed.

## Completed evidence

The exact-source canonical gate passes 1852 cases (three CUDA skips).
The final host scorer/replay and HMM/HSMM preservation bundle passes 134
targeted cases. Nine FINISHED/complete Qlib records retain 99 verified models,
fixed chronological inputs, source hashes and the original fit budget.
Selected HSMM beats HMM on validation and test next-state log loss for all
six nongeometric profiles; geometric controls prefer HMM in all three.
The completed result supersedes this plan's former pending status:
[Nautilus qualification and durable reference](https://github.com/awa-si/nautilus/blob/1bbcd0b2ffd5d5d8766d702c8a0185ae9b351594/docs/usable-hsmm.md).

## Remaining scope

The observations in the qualification generator are well separated. The
preregistered frozen-model noise/shift diagnostic is now complete: all nine
profiles fail at least one diagnostic quality case, with no optimizer updates
or artifact changes. All 72 probability/score audits pass, and all three
state prediction classes remain represented.

Broader observation robustness is not qualified. The separately fixed
[train-only variance study](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-emission-calibration.md)
reduced severe-noise errors but failed its clean HMM floor, so broad variance
was not adopted as a general default. A separate
[causal noise router](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-noise-routing.md)
passed fresh synthetic validation/test gates while leaving the original clean
forecasts unchanged; its scalar Float64 stationary-prior contract does not
admit the retained three-feature Float32 market artifacts. Constant mean shift
remains unresolved. These host studies changed neither the qualified model
core nor a production default.
See the original [frozen-model diagnostic](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-robustness.md)
and the [market admission stop](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-market-routing-admission.md).
Keep already consumed synthetic slices labelled as diagnostics. Any new fitting
or context comparison requires a separate prospective protocol and fresh
admissible evidence. Do not open reserved market cohorts or resume exhausted
market budgets under this plan. Neural/context benefit and market production
readiness remain separate gates; the [current directional decision](https://github.com/awa-si/nautilus/blob/main/docs/nhsmm-directional-decision.md)
stops promotion of the fixed market references under the no-new-data constraint.
