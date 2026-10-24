# Usable supervised HSMM
Status: mathematical implementation verified locally; native calibration and final qualification pending. No new market campaign.
1. Boundary-aware training cohorts: require provenance; trim unknown first episode.
2. Censored Beta-smoothed duration hazards and pooled Dirichlet-smoothed transitions.
3. Independent chronological synthetic calibration versus geometric HMM, with fixed train/validation/test splits.
4. Exact second derivatives of supervised filter score at h_tail=1.
5. Full canonical HMM/HSMM gate, package/default import and native serving verification.
Live inference consumes historical observations without validation-label trimming. An explicit stationary state/age prior is available for time-homogeneous unknown-age entry; reset preserves the caller-selected entry policy.
Existing ordinary APIs keep their fresh-episode contract. No arbitrary initial-age law is guessed.
