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

The observations in that generator are well separated. Before broader usability
claims, preregister a frozen-model diagnostic with weaker observations and
controlled shifts, fixed metrics and no new candidate selection.
Keep already consumed synthetic slices labelled as diagnostics. Do not open
reserved market cohorts or resume exhausted market budgets under this plan.
Neural/context benefit and market production readiness remain separate gates.
