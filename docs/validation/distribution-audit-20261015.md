# Distribution audit — NHSMM regime-v2

Reviewed source: `awa-si/nhsmm@2359bf041456c49f4e0700dfa9b116dfe9759752`.
All distribution implementation code was read: `distributions/__init__.py` and all1,146 lines of `distributions/default.py`. Coupling was traced through DistributionSet, initialization, optimization, causal filtering, tail hazards, artifacts and inference preparation.

**Audit only:** no model implementation fixes, no additional model fitting and no changed frozen holdout scores. Detailed outputs and reviewed scope are recorded in the matching JSON report.

## Confirmed defects

| ID | Priority | Finding | Baseline relevance |
| --- | --- | --- | --- |
| D08 | High | Gaussian and Student-t means are clipped at100,000 by categorical logit validation; joint translation fails and mean gradients vanish. | Material for large raw positive features, not the standardized baseline. |
| D01 | High | Valid exact-zero duration support produces NaN hazard gradients. | Public differentiable kernel; ordinary finite-score baseline does not demonstrate this trigger. |
| D02 | High | Categorical probabilities containing zero produce all-NaN gradients even for an observed positive-probability class. | Probability API, not the default logit-matrix likelihood path. |
| D03 | Medium | Student-t moments/entropy fail to honor parameter broadcasting. | Primitive API; fully expanded model emissions pass reference comparison. |
| D04 | Medium | Finite hard-mask sentinel leaks forbidden probability at saturated logits in Initial, Duration and Transition. | Reproduced with finite Float64 logits -1e40; not observed during the baseline. |
| D05 | Medium | Categorical(validate_args=True) runs validation before _logits exists. | Constructor validation path. |
| D06 | Medium | Scalar labels cannot broadcast over batched Categorical distributions. | Primitive log_prob API. |

D07 is a separate **contract discrepancy**, not a proven likelihood-kernel defect: Duration.log_matrix applies soft_dmax, while expected_probs/forward accept the same keyword but ignore it. Generic diagnostics may therefore inspect a different PMF than the model uses. Decide whether those helpers must honor the gate or reject it explicitly.

### Minimal numerical reproductions

- D08: set Gaussian mu or Student-t loc to[0,1], evaluate x=0; add1e6 to both locations and observation. Effective locations become[100000,100000], likelihoods no longer match and location gradients are[0,0].
- D01: pass log-duration[[[0,-inf,-inf]]] with requires_grad; backpropagate sum(exp(log_end))+sum(exp(log_continue)). All three input gradients are NaN.
- D02: Categorical(probs=[0,.4,.6]).log_prob(1) is finite; backward gives NaN for all three probability entries.
- D03: loc=zeros(2,3), scalar scale=1 and df=5, event_dim=1 declares batch[2] but returns scalar entropy1.62750 instead of[4.88251,4.88251]. Scalar loc with scale=ones(2,3) returns scalar mean rather than shape[2,3].
- D04: with a one-allowed-category mask, all logits -1e40 in Float64 clamp to the same finite value as the forbidden sentinel; masked probabilities become[.5,.5]. A coordinated fix must preserve categorical finite/negative-infinity validation and all-impossible-row rejection.
- D05: Categorical(logits=zeros(3),validate_args=True) raises AttributeError: missing _logits.
- D06: logits shape[2,3] and scalar label1 should return shape[2], as PyTorch does; current expand raises RuntimeError.

## What passed

137 existing distribution/context/emission/duration/filter/transition tests passed in17.79s on CPU.
Additional checks show Gaussian and Student-t manual emission likelihoods agree with returned PyTorch distribution objects in Float32 and Float64, with finite gradients for every active parameter. Maximum observed Gaussian discrepancy was9.54e-7(Float32),8.88e-16(Float64); Student-t was exact in these probes. Positive-support duration hazard gradcheck passes.

Existing tests confirm tail h=1 compatibility, survival beyond D, runtime/batch filtering equivalence, context applied once, parameter identities, support/normalization on ordinary values and tail boundary recovery. CUDA was not available. Passing these tests does not cover the newly reproduced failures.

## Four model components — interpretation

- **Initial:** categorical state axis and canonical causal context route are consistent on ordinary inputs; D04 affects extreme masked support.
- **Transition:** destination normalization and [K,D,K] routing are consistent. Ergodic support deliberately includes self-renewal. This is a model contract, not silently a coding error.
- **Duration:** the last entry is D+ mass with a geometric tail when enabled. D64 is therefore not a hard maximum duration. Tail probability projection is present after optimizer steps, and fresh restarts recreate it. The active PMF combines free duration logits with a trainable log-sigmoid gate; the decomposition is non-identifiable without additional constraints. That is a design/optimization concern, not proof of wrong normalized probabilities.
- **Emission:** both implemented likelihood formulas pass reference checks under ordinary fully expanded inputs. Gaussian covariance is diagonal and Student-t coordinates are independent. Legacy initialization intentionally gives Gaussian effective variance softplus(0)=0.6931 and Student-t effective scale softplus(1)=1.3133; tests explicitly preserve this behavior. These values must not be casually described as unit variance/scale. D08 is an independent location-clipping defect.

## Frozen artifact explains the next research boundary

Artifact SHA256:`925bc1d35721d031d5385e9fa52ffc381eb789e1abfe225d18468878f6b2130b`.

After ten updates, expected model segment durations are[31.23,31.29,31.86,32.28] bars, close to the uniform D64,h=.5 initial mean32.515625. D+ mass is only about1.46–1.75%, with tail end probabilities0.420–0.580. This verifies weak duration adaptation in this run, not a converged estimate.

Training-only complete true-episode means are bull117.62,bear95.90,range104.81,chaotic36.76 bars. Direct comparison needs care: learned boundary transition self-probabilities range approximately0.229–0.281, while the reference generator forbids self transitions. A model episode can reset its age without changing latent regime.

**Correction to earlier metric interpretation:** capped-age MAE in the frozen holdout report is exploratory, not a directly comparable regime-age accuracy measure under these different episode semantics. Do not edit the frozen scores to conceal this limitation.

The training features also have marginal correlations0.778(return/trend) and0.854(volatility pair), while Gaussian emissions are conditionally diagonal. Marginal correlation alone does not prove within-state dependence or establish the cause of oversegmentation.

## Recommended order

1. Correct unintended location clipping and exact-zero-support gradients; add targeted reference and backward regressions.
2. Correct categorical validation/broadcasting and Student-t broadcasted statistics.
3. Restore exact hard support across the primitive/matrix/kernel interfaces.
4. Resolve gate-helper contracts and align episode-age evaluation semantics.
5. Only then preregister training/validation diagnostics of duration adaptation, emissions, age-dependent transition complexity and optimizer convergence. Keep the late test frozen and require a fresh outer evaluation for tuned candidates.

No source fix or distribution redesign is authorized by this audit report itself. The known edge-case defects have not been shown to trigger in the standardized Gaussian baseline. No production or trading promotion follows.
