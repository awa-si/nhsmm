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

**Current status:** pending.

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
