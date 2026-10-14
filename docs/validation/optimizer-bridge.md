# Optimizer bridge — Step35

Step52-1931-joint: verified immutable snapshot; 80 actual updates saved.
Completed phases known_emission_control,estimated_emission_frozen; active estimated_emission_joint at20 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1931-checkpoint-joint-native-20261014.json.


Step52-1931-frozen: verified immutable snapshot; 50 actual updates saved.
Completed phases known_emission_control; active estimated_emission_frozen at20 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1931-checkpoint-frozen-native-20261014.json.


Step52-1931-control: verified immutable snapshot; 25 actual updates saved.
Completed phases ; active known_emission_control at25 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1931-checkpoint-control-native-20261014.json.


Step53-1930-complete:90 actual EM updates (30 per arm), independently verified.
All52 source/software/protocol/data/initialization/history/grid/digest checks,
terminal/outer/public LL/all gradients/segment and causal filter replay pass.
Verification adds0 updates; completed resume is byte-preserving. Oracle OFF,
frozen source ac312d6. Raw report SHA256bcb1aa4def5acf4d844eb79aa8760c29014b65f65e969a8f8ba9b252d597c818.

| Arm | Duration TVs | Max mean error | Max variance relative error | Outer gain nats/row | Full gate |
| --- | --- | ---: | ---: | ---: | --- |
| known_emission_control | 0.072082/0.085470/0.077466/0.085500 | 0.000000 | 0.000000 | 0.009734341 | PASS |
| estimated_emission_frozen | 0.731847/0.940677/0.875870/0.868605 | 0.827676 | 0.658552 | 0.107755687 | FAIL |
| estimated_emission_joint | 0.072055/0.085463/0.077425/0.085414 | 0.010437 | 0.013317 | 0.439628733 | PASS |

Joint minus frozen estimated outer LL2389.485931825 nats.
Each outer gain uses its arm's initial emissions; joint gain includes emission
learning. Known control is a separate information reference. Exposure post-fit:
{"completed_per_state":[655,687,905,814],"terminal_censored_per_state":[344,236,351,93]}. No post-outer tuning.
Evidence: validation/results/joint-gaussian-duration-seed1930-native-20261014.json
and matching verification report. Scalar/static known transition/initial study;
no neural/Trading or legacy NPZ acceptance.


Step52-1930-estimated_emission_joint-20: verified immutable snapshot,
completed phases known_emission_control,estimated_emission_frozen; active estimated_emission_joint at20 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1930-checkpoint-joint-native-20261014.json.

Step52-1930-estimated_emission_frozen-20: verified immutable snapshot,
completed phases known_emission_control; active estimated_emission_frozen at20 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1930-checkpoint-frozen-native-20261014.json.

Step52-1930-control25: immutable control-arm25-update snapshot passes frozen
ac312d6 source/software/contract/data/initialization/history/grid integrity.
Oracle OFF; other arms, outer outcomes and final independent replay pending.
Evidence: results/joint-gaussian-duration-seed1930-checkpoint20-native-20261014.json.
The historical filename is not the update counter; the verified payload is25.

## Step51 — full joint study execution gate passed

Source ac312d6 freezes52 source/software/protocol files. Optimized16x300 runner
smoke performs six actual updates, exact interruption/resume and independent
terminal/outer/public/segment verification in4.602s; all numerical gates pass.
Local three-repeat log-duration E-step versus scaled joint-statistics benchmark:
0.137762s versus0.066650s (2.067x). This is not production/neural speed evidence.
23 focused tests plus static/compilation checks pass. Raw smoke:
results/joint-runner-scaled-smoke-native-20261014.json. The unchanged statistical
contract1930/1931/1932,30 updates per arm (270 total) can execute with atomic
checkpoints and independent final replay. Oracle OFF; recovery outcomes pending.

## Step50 — bounded runner execution and arithmetic performance

Frozen bbfc8df runner smoke performs six actual EM updates on16x300 observations,
including exact interruption/resume and three-arm independent numerical replay;
all gates pass in6.024s. Evidence: results/joint-runner-smoke-native-20261014.json.
Smoke uses an explicitly reduced budget, not the statistical protocol. Prior to
any full fit, joint backward arithmetic is changed to equivalent evidence-scaled
probabilities with finite/conservation guards; the independent old log recursion
is untouched.23 focused tests pass in2.92s, Ruff/Black pass. The model, seeds,
initialization, budget and gates are unchanged. Oracle OFF. Optimized-source
bounded smoke and full frozen study execution remain pending.

## Step49 — frozen joint statistical runner

The three-arm runner implements30 EM updates per arm from uniform full positive
duration PMFs. Estimated frozen/joint emissions share training-only initialization;
known emissions are a separate information control. Atomic0/every5/final saves
contain exact PMF/means/variances/history and state digest. Resume checks fixed
source/software/protocol/initialization, grid/history and terminal likelihood.
Independent verification replays terminal/outer LL, all parameter gradients,
public and segment causal filtering, and completed byte-preserving no-op without
new updates. 23 focused tests pass (3.42s); Ruff passes. Oracle OFF. Bounded actual
runner smoke precedes fresh seed1930/1931/1932 fits; no results or gates yet.

## Step48 — bounded joint execution confirmed

Frozen source666013d:16x300 observations, three actual joint EM updates in0.782s.
Augmented marginal objective improves monotonically from -12625.720454 to
-10379.349808. Both source hashes, regenerated training identity, counters and
finite monotonic scores independently verified. Raw evidence is
results/joint-gaussian-duration-smoke-native-20261014.json. Oracle OFF and no
outer evaluation. Observation-only initialization and joint soft-posterior
updates execute; statistical joint recovery is still untested. Next boundary is
the checkpointed runner and independent replay for the frozen three-arm protocol.

## Step47 — free scalar Gaussian emissions

Research joint_gaussian_duration computes soft state occupancy and weighted
observation moments alongside censored duration posterior counts. Its M-step
updates Gaussian means/variances and the existing positive fixed-bin family.
All21 focused tests pass in2.66s: public LL/all parameter gradients1e-7,
causal filter1e-8, conservation/chunk checks, monotone updates and validation.
No prior experiment owner or production optimizer is changed. Oracle OFF.
The preregistered joint-gaussian-duration-protocol.json compares known-emission
control, frozen observation-initialized emissions and jointly learned emissions.
Known control is an information reference, not initialization-matched; joint
versus frozen estimated arms share initialization. Static transition/initial
parameters remain known. Fresh statistical fits and checkpointed runner pending.
The upcoming16x300 smoke establishes execution only, not statistical recovery.

## Step46 — matched recovery batch complete

Step46 completes all frozen matched fixed-bin optimizer recovery seeds1922/1923/1924.
Each uses1024x300 training and24 outer sequences, shared EM warm3 and27 updates
per matched arm. Total171 updates (90 EM M-steps,81 Adam steps); verification
adds0 updates. All23 source/software/contract/data/history/grid/Adam-moment checks,
independent terminal/public LL/gradient/filtering replays and completed no-ops pass.
EM3/3 and Adam3/3 pass per-state TV<=0.12 and positive heldout gain. EM has a
slightly higher outer LL on3/3 seeds; this bounded27-update result is not an
equal-compute or global convergence claim. Both optimization methods recover
durations in this known-nuisance fixed-family study. Oracle OFF, source bd8e3f6;
all raw reports/proofs are SHA-bound. No protocol tuning follows outer outcomes.
Next: separately validate learning emission parameters alongside durations,
then transition/initial parameters and neural context. Keep representation,
initialization, exposure and optimizer effects separate. Current evidence does
not establish joint-neural/Trading or legacy NPZ acceptance.

| Seed | Arm | TV bull/bear/range/chaotic | Outer gain nats/row | Gate |
| --- | --- | --- | ---: | --- |
| 1922 | fixed_bin_em | 0.091594/0.102778/0.087889/0.076981 | 0.006630394 | PASS |
| 1922 | fixed_bin_adam | 0.086533/0.103746/0.091118/0.077992 | 0.006586749 | PASS |
| 1923 | fixed_bin_em | 0.075270/0.086766/0.078203/0.064095 | 0.006388083 | PASS |
| 1923 | fixed_bin_adam | 0.072777/0.088392/0.080989/0.065023 | 0.006375065 | PASS |
| 1924 | fixed_bin_em | 0.067148/0.091431/0.084801/0.079921 | 0.008825200 | PASS |
| 1924 | fixed_bin_adam | 0.063017/0.091652/0.087694/0.079995 | 0.008773673 | PASS |

Seed1922: Adam minus EM outer LL-0.314248339 nats; final training objective difference-12.716737197 nats; completed episodes661/718/943/845.
Seed1923: Adam minus EM outer LL-0.093727031 nats; final training objective difference-12.854601547 nats; completed episodes686/660/877/836.
Seed1924: Adam minus EM outer LL-0.370991264 nats; final training objective difference-12.448160311 nats; completed episodes657/689/876/819.

Evidence: validation/results/fixed-bin-optimizer-batch-native-20261014.json.
The21.31x score benchmark is a separate matched local frozen-nuisance timing,
not a measured end-to-end speedup of this full study or the neural optimizer.

Step45-1924-complete

Seed1924 completes57 actual updates (30 EM M-steps,27 Adam steps). All23 source/software/contract/training/outer/phase/history/grid/moment checks and independent terminal/public LL/gradient/filtering replays pass; verification performs0 updates and completed resume is byte-preserving. Frozen source bd8e3f6, oracle OFF. Raw report SHA256ebf0c46646848c378bbcec2a563f84dcd62c0ba1c59dd958dcf6f87441440d47.
fixed_bin_em: TVs[0.06714753861712444,0.09143066816603171,0.08480119802051986,0.07992104750851454], outer gain0.008825199756440472 nats/row, full gatePASS.
fixed_bin_adam: TVs[0.06301651551817443,0.09165226525236805,0.08769441043826888,0.07999485259912463], outer gain0.00877367319205051 nats/row, full gatePASS.
Adam minus EM outer LL-0.3709912636077206 nats. Post-fit completed/censored exposure{"completed_per_state":[657,689,876,819],"terminal_censored_per_state":[354,209,372,89]}. Known-nuisance duration-only evidence; no joint-neural/Trading acceptance. No post-outer tuning.

Step45-1924-fixed_bin_adam-20

Seed1924 verified checkpoint: fixed_bin_adam at20 updates; complete phases warm,fixed_bin_em. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1924-fixed_bin_adam-0

Seed1924 verified checkpoint: fixed_bin_adam at0 updates; complete phases warm,fixed_bin_em. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1924-fixed_bin_em-0

Seed1924 verified checkpoint: fixed_bin_em at0 updates; complete phases warm. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1923-complete

Seed1923 completes57 actual updates (30 EM M-steps,27 Adam steps). All23 source/software/contract/training/outer/phase/history/grid/moment checks and independent terminal/public LL/gradient/filtering replays pass; verification performs0 updates and completed resume is byte-preserving. Frozen source bd8e3f6, oracle OFF. Raw report SHA256adb0e8b5411d401bfd412634bece6867e44efa979b5da2e09932e8481111d612.
fixed_bin_em: TVs[0.0752702945870401,0.08676611376649788,0.07820326789831794,0.06409483572621358], outer gain0.006388082791065951 nats/row, full gatePASS.
fixed_bin_adam: TVs[0.07277652286599537,0.08839222894485098,0.08098915628330385,0.06502311069380386], outer gain0.0063750651478428864 nats/row, full gatePASS.
Adam minus EM outer LL-0.09372703120607184 nats. Post-fit completed/censored exposure{"completed_per_state":[686,660,877,836],"terminal_censored_per_state":[332,217,370,105]}. Known-nuisance duration-only evidence; no joint-neural/Trading acceptance. No post-outer tuning.

Step45-1923-fixed_bin_adam-20

Seed1923 verified checkpoint: fixed_bin_adam at20 updates; complete phases warm,fixed_bin_em. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1923-fixed_bin_adam-0

Seed1923 verified checkpoint: fixed_bin_adam at0 updates; complete phases warm,fixed_bin_em. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1923-fixed_bin_em-0

Seed1923 verified checkpoint: fixed_bin_em at0 updates; complete phases warm. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1922-complete

Seed1922 completes57 actual updates (30 EM M-steps,27 Adam steps). All23 source/software/contract/training/outer/phase/history/grid/moment checks and independent terminal/public LL/gradient/filtering replays pass; verification performs0 updates and completed resume is byte-preserving. Frozen source bd8e3f6, oracle OFF. Raw report SHA2560d98e942e65381039e040072cc8f4ad48b40ea381d1d170e1fecbeab9e489d4f.
fixed_bin_em: TVs[0.09159353053815715,0.10277799092806991,0.08788893329491097,0.07698074681814517], outer gain0.0066303944534992195 nats/row, full gatePASS.
fixed_bin_adam: TVs[0.08653307346598754,0.1037455118321417,0.09111849928619238,0.07799179374904616], outer gain0.006586748850816851 nats/row, full gatePASS.
Adam minus EM outer LL-0.3142483393130533 nats. Post-fit completed/censored exposure{"completed_per_state":[661,718,943,845],"terminal_censored_per_state":[352,235,331,106]}. Known-nuisance duration-only evidence; no joint-neural/Trading acceptance. No post-outer tuning.

Step45-1922-fixed_bin_adam-15

Seed1922 verified checkpoint: fixed_bin_adam at15 updates; complete phases warm,fixed_bin_em. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1922-fixed_bin_adam-0

Seed1922 verified checkpoint: fixed_bin_adam at0 updates; complete phases warm,fixed_bin_em. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

Step45-1922-fixed_bin_em-5

Seed1922 verified checkpoint: fixed_bin_em at5 updates; complete phases warm. All23 source/software/contract/training identities, history/grid/PMF and Adam moment/step guards pass. Final outer/replay/recovery gates remain pending. Frozen source bd8e3f6, oracle OFF. Resume only with exact saved runner source and optimizer state.

## Steps42–43 — exact posterior score and matched timing

For known fixed emissions/transitions/initial parameters at temperature1,
the marginal duration gradient equals the expected censored-episode count
score. Aggregate expected counts into the fixed bins, subtract the state total
times bin probability mass, then add the age prior and any declared curvature
derivative. The final bin log density is anchored0. No labels or true support
enter this calculation. This identity does not remove neural/nuisance gradients
when those parameters are trainable.

Six tests pass, including independent concave-count Autograd gradients and
native public censored-likelihood/Adam-step agreement. On the exact same
4x300 fresh401921 observations,uniform initial bins,prior1e-6,clip5,lr0.01,
three timed trials give medians1.057941s public Autograd and0.049636s analytic
score (21.31x). Objective/gradient/step discrepancies are respectively
1.818989e-12,4.463097e-14,3.469447e-18. Setup is excluded; score/backward and
one clipped Adam update are included. No outer/statistical model selection.

The next statistical comparison is separately predeclared in
fixed-bin-optimizer-recovery-protocol.json. It matches both optimizer arms on
full positive width8 density bins,same1024x300 training data,same shared EM
warm3 start and same target/prior;27 updates per arm are a bounded comparison,
not equal-compute or convergence guarantees. Fresh1922/1923/1924 outer outcomes
must not change protocol choices. Exact checkpointed Adam moments/step count
and independent final replay are required before accepting results. Runner and
study execution remain pending. Oracle OFF.

## Steps40–41 — accumulation implemented and verified

Opt-in optimizer_batch_size bounds the number of whole sequences whose graph
is retained for backward. Sum all likelihood gradients, apply the global bias
penalty once, then clip/step once per full training batch. Target monitoring and
final restart scores sum all chunks. Deterministic loss, gradient and two-step
Adam comparisons pass for plain float64 and context/encoder float32 models,
including uneven chunks and variable/empty sequences. Full inputs and
initialization storage are not reduced by this graph bound.

All195 focused cases pass. At fresh seed401919,16x300 observations, batch4,
one actual Adam update takes6.174489s and peaks at582652 KiB (568.996 MiB);
terminal fixed-target/reference likelihood error is0. Known frozen nuisance is
preserved and there is no outer evaluation. The earlier4x300 process peak was
581704 KiB: these measurements support the practical bounded-batch path,
but differ in seed/source/run and are not an equal-condition speed benchmark
or a guarantee about full1024-sequence resource use.

Next: freeze a sufficient-exposure statistical optimizer comparison before
outcomes, match duration family/initialization/data/target and record actual
compute. Development checks may be batched; training remains concurrency1.

## Step39 —300-row resource smoke completed

The fixed fresh401918 training-only smoke runs the actual public optimizer on
4x300 rows, one constant-temperature Adam update at lr0.01, full positive
duration support and frozen known nuisance/current initialization. It takes
1.795389s in the observed native environment; the whole process peaks at
581704 KiB (568.07 MiB). Terminal target/public/reference LL error is
9.094947017729282e-13. It uses no held-out data or labels to choose a model.
Source d2d2389a and exact data/core/runner hashes are recorded in the raw report.

Next resource boundary is exact gradient accumulation over bounded sequence
chunks: sum the sequence likelihood gradients, then clip and take one Adam
step per full batch. Test against unchunked float64 loss/gradients/updates on a
small problem, including uneven final chunks and variable lengths. Fixed-target
monitoring must also aggregate over the full training sample. A full-size fit
requires a separately frozen statistical protocol and measured bounded memory;
this single4-sequence timing does not justify linear memory extrapolation.

## Step38 — pilot complete and independently verified

Source21428b6f, fresh seeds1915/1916/1917,4 training and2 outer sequences of60
rows, sigma2,2 updates per arm,18 total updates. All21 source identities and
data hashes, terminal training/public target likelihoods, independent outer
LL/gradient/filter replays and byte-preserving completed CLI no-op pass.
The verifier performs0 fitting updates and validates saved endpoints; it does
not reproduce the optimizer trajectory. Frozen nuisance preservation is
checked during the fits and its certificate is checked during verification.

| Seed | Constant Adam gain/row | Annealed Adam gain/row | EM gain/row |
| --- | ---: | ---: | ---: |
| 1915 | +0.000148862 | +0.000148224 | +0.003463649 |
| 1916 | +0.000114517 | +0.000114034 | -0.094563277 |
| 1917 | +0.000077814 | +0.000077500 | +0.002064989 |

All three arms fail every seed's full recovery gate (TV<=0.12 in each state
and positive outer gain). Constant Adam has a slightly larger outer gain than
annealed Adam on these three runs, but the tiny budget/sample does not establish
statistical superiority, convergence, equal compute, or a cause of earlier
recovery failures. No arm or hyperparameter is selected from these outcomes.

### Post-fit ground-truth exposure audit

Training labels are used only after completed fits to count observed completed
episodes, treating each final run as censored.

| Seed | Completed bull/bear/range/chaotic | Terminal-censored bull/bear/range/chaotic |
| --- | --- | --- |
| 1915 | 0/0/1/1 | 2/1/1/0 |
| 1916 | 0/0/1/0 | 2/1/1/0 |
| 1917 | 0/0/1/0 | 2/1/1/0 |

Under the configured ground truth, P(duration>60) is
[0.9830170255,0.7730295881,0.5688167179,0.0888006994].
These short sequences provide very little completed-episode evidence for long
durations. Censoring is handled by the likelihood; correct censoring does not
create missing information. These counts do not prove mathematical
non-identifiability, but make the2-step pilot unsuitable for recovery claims.

Next execution gate: measure a300-row actual public optimizer smoke before
allocating a larger-exposure study. Then freeze fresh data/budgets/target model
family and optimizer comparisons before outcomes; compare matching duration
families and report actual compute. Free emissions/transitions/initial and
neural context only after duration-only validation. Oracle remains OFF.

The completed fixed-grid study is accepted only for duration learning with
known nuisance parameters. Both unsmoothed and smoothed arms pass on3/3 seeds;
smoothing improves matched outer LL on3/3 seeds, but does not improve every
state TV or every mean RMSE. These observations do not prove a universal
smoothing benefit, a sample-size effect, or joint neural recovery.

## Bounded temperature diagnostic (2026-10-14)

No fitting, no labels, full positive configured ages1..260, no true-law support
mask. Generate2 sequences of40 observations at sigma2 with seed401915;
use a uniform duration PMF and the existing known-nuisance public model helper.
The helper retains an all-true support buffer; it excludes no configured age.

At the same unchanged parameters, public `log_likelihood` and `_compute_loss`
with both temperature bounds1 and loss_bias0 return exactly
`-177.68856980552732` (absolute difference0).
With `_compute_loss`'s default schedule, iterations0/1/2 of a3-step budget
return `[-177.6885917737634, -177.68913551351594, -177.69322209619298]`.
The fixed-parameter LL span is0.0046303224295911605.
The two regression tests also verify state tensors remain unchanged.

This establishes an objective mismatch, not a faulty likelihood recursion:
temperature divides distribution logits and changes the model being scored.
`NHSMM.optimize` calls this scheduled objective for both updates and final
restart scoring. Thus convergence traces and restart scores are not generally
the public temperature1 likelihood. This diagnostic alone does not establish
that annealing causes the observed statistical recovery failures.

## Next bounded implementation boundary

1. Expose an explicit temperature1 optimizer control, retaining existing
   annealing as a named option. Test config validation and default compatibility.
2. Score final restart snapshots using the declared target temperature, so
   traces identify scheduled versus target objectives explicitly. Do not remove
   the causal encoder or change censoring/runtime semantics.
3. Before any long fit, verify the actual optimize path (including initialization,
   frozen parameters, final snapshot and restart selection), then preregister a
   fresh duration-only optimizer comparison. Match data, initialization and target
   model family; Adam steps and EM M-steps are not equivalent compute budgets.
4. Only after that comparison, progressively free emissions, transitions and
   initial probabilities, then neural context. Report each boundary separately;
   do not jump from known-nuisance EM to Trading acceptance.

Step35 does not alter production optimizer defaults, the frozen study source,
raw outcomes or gates; no further long study has started. Oracle remains OFF.
