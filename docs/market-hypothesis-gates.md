# HSMM market-hypothesis gates

This milestone begins only after the package-core correctness milestone (Gates 0–10) is accepted. It tests whether explicit-duration latent regimes describe real market structure and add out-of-sample information. It is not part of the mathematical correctness claim.

The gates are deliberately falsifiable. A negative result is an accepted outcome and must not be repaired by weakening thresholds after observing test results.

## Gate 11 — Non-geometric regime duration structure

**Hypothesis:** real inferred market regimes exhibit persistence that is not adequately explained by the geometric-duration assumption of an ordinary HMM.

Required evidence:

- fit comparable HMM and HSMM regime models on the same training windows and features;
- estimate empirical run-length / survival behavior on strictly held-out windows;
- compare state-conditional persistence against the fitted geometric HMM baseline;
- measure whether exit hazard changes materially with episode age;
- repeat across independent walk-forward splits and relevant market regimes;
- report uncertainty and effective sample counts per state/age bucket.

**Acceptance:** predeclared OOS tests reject the geometric-duration baseline for a material subset of stable states/splits, with replicated direction and effect size. Evidence must survive state-label alignment and reasonable state-count sensitivity.

**Failure meaning:** if duration is effectively geometric or unstable OOS, explicit-duration structure is not justified for the trading application.

**Frozen first-pass contract (2026-10-07):**

- market/data: Binance BTCUSDT perpetual, completed Nautilus daily catalogs only;
- usable interval: 2023-10-03 through 2023-11-25; incomplete `2023-11-26.building` is excluded;
- decision cadence: 5-minute trigger;
- observation contract: `axis-observation-v2-to-nautilus-temporal-observations-v1`, 18 coordinates;
- causal equal-timestamp event order: `1m -> 5m -> 15m -> 1h`;
- state count: `K=3`;
- emissions: Gaussian;
- initialization: `emission_init_mode="kmeans"`;
- internal ContextEncoder: disabled;
- optimization: package default `n_init=3`, `max_iter=40`;
- preprocessing: feature-wise mean/std fitted on training observations only and then applied unchanged to OOS;
- sequence unit: UTC daily sequences; model state resets at each day boundary for both train and OOS, identically for HMM and HSMM;
- HMM baseline: causal NHSMM recurrence with `max_duration=1`;
- explicit-duration candidate: causal NHSMM with `max_duration=96` (8 hours at the 5-minute cadence);
- W1 train: 2023-10-03..2023-10-20, OOS: 2023-10-21..2023-10-31;
- W2 train: 2023-10-03..2023-10-31, OOS: 2023-11-01..2023-11-12;
- W3 train: 2023-10-03..2023-11-12, OOS: 2023-11-13..2023-11-25;
- model seeds by split: W1=1101, W2=1102, W3=1103.

Primary OOS metric is per-timestep log-likelihood gain:

```text
delta_ll = HSMM_oos_log_likelihood_per_step - HMM_oos_log_likelihood_per_step
```

A split has a material OOS duration advantage when `delta_ll >= 0.01` nats/timestep. Replication requires at least 2 of 3 splits to meet that bound and median `delta_ll > 0`.

The mechanism guard evaluates the learned context-free HSMM duration hazard only over materially supported ages: ages whose survival probability is at least `0.10`. For states with fitted training occupancy at least `0.10`, a material non-geometric duration effect requires `max(hazard)-min(hazard) >= 0.10`. At least one materially occupied state must meet this condition in at least 2 of 3 splits.

Gate 11 first-pass acceptance requires **both** replicated OOS likelihood advantage and replicated material hazard non-constancy. If the K=3 first pass passes, state-count sensitivity at K=2 and K=4 is confirmatory work required before final Gate-11 closure. Thresholds above remain frozen for that confirmation.

**Current status:** running under the frozen first-pass contract.

## Gate 12 — Predictive state separation

**Hypothesis:** inferred latent state carries information about the distribution of future market outcomes.

Evaluate future outcomes without using them during state inference. At minimum examine:

- forward return distribution;
- realized volatility / range;
- directional persistence or reversal;
- activity / participation / flow variables where available;
- transition or boundary-conditioned behavior.

Required controls:

- OOS-only outcome evaluation;
- no look-ahead in features, context, state posterior, or alignment;
- posterior-weighted and hard-state views;
- state-frequency / occupancy reporting;
- multiple horizons and walk-forward splits with multiplicity-aware interpretation.

**Acceptance:** at least one predeclared future-outcome family shows replicated, economically non-trivial state separation OOS, with the sign/ordering stable enough to survive adjacent splits.

**Failure meaning:** if states are statistically distinguishable only in contemporaneous features but not future outcomes, they do not yet support a predictive trading-regime claim.

**Current status:** pending.

## Gate 13 — Incremental episode-age information

**Hypothesis:** conditional on latent state, episode age adds predictive information. This is the application-level reason to prefer an HSMM over a state-only HMM.

Required comparisons:

- state-only predictor versus state + age;
- state posterior versus joint state-age posterior;
- optional hazard / survival forecast features;
- matched OOS windows and identical downstream target definitions;
- age-bucket reliability and minimum-support guards.

Evaluate whether age changes:

- probability of state persistence / boundary;
- future return distribution;
- future volatility / range;
- other predeclared downstream outcomes.

**Acceptance:** age or survival information produces replicated incremental OOS information beyond state alone, under a predeclared effect-size/statistical threshold and without relying on tiny-support age tails.

**Failure meaning:** if age adds no stable information beyond state, the explicit-duration HSMM mechanism is not justified by downstream utility even if Gate 11 detects non-geometric durations.

**Current status:** pending.

## Gate 14 — Out-of-sample utility against simpler baselines

**Hypothesis:** the accepted NHSMM state/age information improves downstream prediction or decision quality relative to simpler alternatives.

Required baselines:

- no-regime / raw-feature baseline;
- ordinary HMM or equivalent geometric-duration latent-state baseline;
- state-only NHSMM ablation;
- state + age/posterior NHSMM;
- optionally survival/hazard-enhanced NHSMM when Gate 13 supports it.

Evaluation must use frozen train/OOS boundaries and identical downstream labels, costs, and execution assumptions. Report both predictive and trading-facing metrics; trading metrics cannot substitute for predictive diagnostics.

At minimum record:

- OOS likelihood / calibration where applicable;
- predictive discrimination or error;
- stability across walk-forward splits;
- turnover and signal frequency;
- net performance after declared fees/slippage for trading experiments;
- degradation under reasonable cost and latency stress.

**Acceptance:** the NHSMM representation shows replicated OOS improvement over both the no-regime and geometric-duration baselines. Any trading claim additionally requires net utility after declared execution costs and must survive adjacent walk-forward periods.

**Failure meaning:** a mathematically correct and statistically identifiable NHSMM can still fail this gate. Such a failure rejects the current trading-usefulness hypothesis, not the package core.

**Current status:** pending.

## Execution order and stop conditions

Run Gate 11 first, then Gate 12, then Gate 13. Gate 14 is the final application-utility gate.

Do not use Gate 14 profitability to retroactively justify a failed Gate 11–13 mechanism claim. Conversely, passing Gates 11–13 does not imply profitable trading.

Every gate must freeze before execution:

- dataset and market universe;
- time range and walk-forward splits;
- feature contract and causal timing;
- state count / model-selection policy;
- target horizons;
- metrics and acceptance thresholds;
- random seeds where relevant;
- baseline definitions;
- transaction-cost assumptions for trading-facing evaluation.

The narrow completion claim after Gates 11–14 is:

> On the maintained real-market evaluation, explicit-duration latent state and episode-age information provide replicated out-of-sample information beyond simpler state-only and no-regime baselines under the frozen evaluation contract.

Anything stronger requires separate evidence.
