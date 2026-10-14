# NHSMM State

Current package readiness and next package-level boundary. Detailed model semantics are in [model.md](model.md), configuration and tuning in [configuration.md](configuration.md), validation semantics and accepted package evidence in [validation.md](validation.md), verification policy in [testing.md](testing.md), and release procedure in [release.md](release.md).

## Stepwise repair — current checkpoint

Step61-1935-frozen: immutable checkpoint verified; 60 actual updates saved.
Completed phases known_chain_reference,estimated_chain_frozen; active estimated_chain_joint at0 updates.
60 source/software/protocol/data/initialization/history/grid/all-five state checks pass.
Frozen2294c6e, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-static-chain-seed1935-checkpoint-frozen-native-20261014.json.


Step61-1935-first: immutable checkpoint verified; 20 actual updates saved.
Completed phases ; active known_chain_reference at20 updates.
60 source/software/protocol/data/initialization/history/grid/all-five state checks pass.
Frozen2294c6e, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-static-chain-seed1935-checkpoint-first-native-20261014.json.


Step62-1934-complete:90 actual updates,30 per arm; independently verified.
60 source/software/protocol/data/initialization/history/grid/all-five state checks,
terminal/outer LL, all gradients, segment/filter replay and byte-preserving
completed resume pass. Verification adds0 updates. Frozen2294c6e, Oracle OFF.
Raw report SHA2564b41d738f064672d57e4dcfbd24c3ca8354a2fef1a15e4fba271f2ab03391480.

| Arm | Max duration TV | Max transition row TV | Initial TV | Outer gain nats/row | Full gate |
| --- | ---: | ---: | ---: | ---: | --- |
| known_chain_reference | 0.092252 | 0.000000 | 0.000000 | 0.427448301 | PASS |
| estimated_chain_frozen | 0.092138 | 0.266667 | 0.000000 | 0.425408070 | FAIL |
| estimated_chain_joint | 0.092241 | 0.030540 | 0.006985 | 0.425726570 | PASS |

Per-component gates: {"known_chain_reference":{"duration":true,"mean":true,"variance":true,"transition":true,"initial":true,"outer_gain":true},"estimated_chain_frozen":{"duration":true,"mean":true,"variance":true,"transition":false,"initial":true,"outer_gain":true},"estimated_chain_joint":{"duration":true,"mean":true,"variance":true,"transition":true,"initial":true,"outer_gain":true}}.
Joint minus frozen outer LL2.293196143 nats.
Every arm learns emissions/durations; each gain uses its complete initial model,
including emission learning. Known reference changes chain information. Exposure:
{"completed_per_state":[627,731,870,836],"terminal_censored_per_state":[339,219,378,88]}. No post-outer tuning.
Evidence: validation/results/joint-static-chain-seed1934-native-20261014.json and matching proof.
Scalar/static study; no neural/Trading/legacy NPZ or convergence acceptance.


Step61-1934-joint: immutable checkpoint verified; 80 actual updates saved.
Completed phases known_chain_reference,estimated_chain_frozen; active estimated_chain_joint at20 updates.
60 source/software/protocol/data/initialization/history/grid/all-five state checks pass.
Frozen2294c6e, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-static-chain-seed1934-checkpoint-joint-native-20261014.json.


Step61-1934-first: immutable checkpoint verified; 45 actual updates saved.
Completed phases known_chain_reference; active estimated_chain_frozen at15 updates.
60 source/software/protocol/data/initialization/history/grid/all-five state checks pass.
Frozen2294c6e, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-static-chain-seed1934-checkpoint-first-native-20261014.json.


Step60 passes actual static-chain runner smoke at frozen2294c6e:16 training/
4 outer sequences x300, six updates, interruption/resume and all three public
all-five gradient/segment/filter/independent likelihood gates in6.444s. This
explicitly reduced execution-only contract does not alter statistical protocol.
60 source/software/protocol identities frozen;34 focused tests/static checks pass.
Full1934/1935/1936 study may execute270 updates with atomic checkpoints and
independent final proofs. Oracle OFF; no NPZ fixture dependency or acceptance.

Step59 implements the preregistered static-chain runner and independent final
replay without changing the protocol or prior source owners. Checkpoints include
PMF/means/variances/transition/initial probabilities and full monotone history;
exact five-group resume, frozen-chain guards and terminal replay are tested.
34 focused cases pass in3.34s (14 new runner cases), Ruff/Black/compilation pass.
60 dependency/protocol source paths are frozen before bounded actual runner smoke.
Next: smoke gate, then270 updates on1934/1935/1936 with separate component gates.
Oracle OFF; legacy binary-reference gap remains separate from this synthetic study.

Step58 broad validation at f48c411: complete code/test scope materialized and
208 file identities verified (124 missing code/tests plus two JSON references
restored). Available CPU suite passes1032 cases in41.50s, with3 CUDA skips and
one explicitly excluded unavailable binary-reference case. Initial full run:
1031 pass,2 missing-fixture failures,3 skips; restoring JSON references resolves
the compact-runner failure. Ruff, Black(all203 Python files) and compileall pass.
New all-static smoke independently replayed on all16 terminal sequences and
public all-five gradients/segment/filtering;58 source hashes and data match.
Existing Gaussian and optimizer batch/proof joins and source identities pass.
Legacy NPZ transport remains blocked; candidate reconstruction differs from
the expected Git/SHA256 identity and was not substituted. No numerical drift
is inferred from archive bytes alone. Oracle OFF. New chain statistical fits
remain unexecuted. Evidence: validation/results/all-validation-native-20261014.json.

Step57 confirms actual all-static EM execution at frozen source7ffc718:
16x300 observations, three updates jointly learning duration, Gaussian mean/
variance, transition and initial probabilities in2.208s. Objective increases
from -13164.073646 to -10366.226227. All public gradient/LL/segment/filter checks
pass; independent old-log terminal LL replay on all16 sequences has error0.
58 source/software/protocol hashes and regenerated data/counters verified.
Oracle OFF, full positive duration/allowed-chain support retained; no outer
evaluation or statistical acceptance. Evidence: validation/results/
joint-static-chain-smoke-native-20261014.json.17 focused tests pass.
Next: checkpointed three-arm runner and independent final replay for the
predeclared1934/1935/1936 statistical study. No neural/Trading acceptance.

Step56 adds a separate all-static chain EM research owner: soft expected
boundary transition counts and initial-state posteriors, alongside Gaussian
and censored full-age duration statistics. The endpoint has no outgoing event;
one-row, count-conservation and unchanged known-chain equivalence checks pass.
All17 focused tests pass in2.16s, including public gradients for all five groups,
public/segment causal filtering, monotone augmented EM and invalid chain laws.
Ruff/Black/compilation pass. Oracle OFF; previous study owners unchanged.
The fresh1934/1935/1936 three-arm protocol is preregistered, not executed.
Next: frozen-source16x300 actual all-five-parameter smoke, then statistical runner.

Step55 completes and independently verifies the joint Gaussian/duration batch:
seeds1930/1931/1932,270 actual EM updates,0 verification updates. Full gates:
known control3/3, estimated frozen0/3, estimated joint3/3. All nine numerical
replays, monotone histories and byte-preserving completed resumes pass. Raw
reports/proofs and54-file static closure audit are SHA-bound; frozen ac312d6,
Oracle OFF, protocol unchanged. Joint beats frozen outer LL on3/3 seeds.
Joint max TV0.100770, mean error0.017221, variance relative error0.016725.
Bad fixed observation-initialized emissions can destroy duration recovery even
with correct recursion; learning emissions restores it in this controlled study.
See validation/joint-gaussian-duration-findings.md and the batch evidence.
Next: separately free transition/initial parameters before neural context.
No production default change, Trading/neural/legacy NPZ or convergence acceptance.

Step53-1932-complete:90 actual EM updates (30 per arm), independently verified.
All52 source/software/protocol/data/initialization/history/grid/digest checks,
terminal/outer/public LL/all gradients/segment and causal filter replay pass.
Verification adds0 updates; completed resume is byte-preserving. Oracle OFF,
frozen source ac312d6. Raw report SHA256076112148c9f0a56875d494d051db7671cd16a872ca0c1178f8dc66c620176e2.

| Arm | Duration TVs | Max mean error | Max variance relative error | Outer gain nats/row | Full gate |
| --- | --- | ---: | ---: | ---: | --- |
| known_emission_control | 0.085944/0.076398/0.078658/0.074570 | 0.000000 | 0.000000 | 0.006849968 | PASS |
| estimated_emission_frozen | 0.796726/0.958354/0.895260/0.876619 | 0.922510 | 0.655120 | 0.105160623 | FAIL |
| estimated_emission_joint | 0.085879/0.076343/0.078509/0.074651 | 0.017221 | 0.012487 | 0.461024426 | PASS |

Joint minus frozen estimated outer LL2562.219381083 nats.
Each outer gain uses its arm's initial emissions; joint gain includes emission
learning. Known control is a separate information reference. Exposure post-fit:
{"completed_per_state":[664,677,916,800],"terminal_censored_per_state":[352,236,356,80]}. No post-outer tuning.
Evidence: validation/results/joint-gaussian-duration-seed1932-native-20261014.json
and matching verification report. Scalar/static known transition/initial study;
no neural/Trading or legacy NPZ acceptance.


Step52-1932-joint: verified immutable snapshot; 75 actual updates saved.
Completed phases known_emission_control,estimated_emission_frozen; active estimated_emission_joint at15 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1932-checkpoint-joint-native-20261014.json.


Step52-1932-frozen: verified immutable snapshot; 50 actual updates saved.
Completed phases known_emission_control; active estimated_emission_frozen at20 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1932-checkpoint-frozen-native-20261014.json.


Step54 supplementary static closure audit verifies54 files against frozen Git\nblob identities and SHA256s, including the52 declared manifest paths plus\nnhsmm/benchmark.py and scripts/validate_trading_regime_usefulness.py. All match\nsource ac312d6; no model, source owner, protocol or report changes. The batch\naggregator joins this SHA-bound audit with all three final proofs. Black/Ruff pass.\nEvidence: validation/results/joint-source-closure-audit-native-20261014.json.\nOracle OFF; final third-seed recovery/replay remains pending.\n

Step52-1932-control: verified immutable snapshot; 25 actual updates saved.
Completed phases ; active known_emission_control at25 updates.
All52 source/software/protocol/data/initialization/history/grid/digest checks pass.
Frozen ac312d6, Oracle OFF. Final outer/recovery/public replay remains pending.
Evidence: validation/results/joint-gaussian-duration-seed1932-checkpoint-control-native-20261014.json.


Step53-1931-complete:90 actual EM updates (30 per arm), independently verified.
All52 source/software/protocol/data/initialization/history/grid/digest checks,
terminal/outer/public LL/all gradients/segment and causal filter replay pass.
Verification adds0 updates; completed resume is byte-preserving. Oracle OFF,
frozen source ac312d6. Raw report SHA256dc0a39e96478a86cde3ba9692fcfb54470fc93c08f9c0ef50d638a991ea27ecf.

| Arm | Duration TVs | Max mean error | Max variance relative error | Outer gain nats/row | Full gate |
| --- | --- | ---: | ---: | ---: | --- |
| known_emission_control | 0.076377/0.083206/0.100810/0.075427 | 0.000000 | 0.000000 | 0.007179008 | PASS |
| estimated_emission_frozen | 0.783332/0.952183/0.889545/0.873360 | 0.900280 | 0.653902 | 0.114029613 | FAIL |
| estimated_emission_joint | 0.076391/0.083220/0.100770/0.075376 | 0.008760 | 0.016725 | 0.438753430 | PASS |

Joint minus frozen estimated outer LL2338.011481078 nats.
Each outer gain uses its arm's initial emissions; joint gain includes emission
learning. Known control is a separate information reference. Exposure post-fit:
{"completed_per_state":[676,688,937,805],"terminal_censored_per_state":[355,230,351,88]}. No post-outer tuning.
Evidence: validation/results/joint-gaussian-duration-seed1931-native-20261014.json
and matching verification report. Scalar/static known transition/initial study;
no neural/Trading or legacy NPZ acceptance.


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

Step52-1930-control25: seed1930 known_emission_control at25 actual updates,
immutable snapshot verified against52 frozen source/software/protocol files,
training hash, observation initialization, model/history/grid and state digest.
Source ac312d6, Oracle OFF. Other arms and final outer/recovery replay pending.
Snapshot: validation/results/joint-gaussian-duration-seed1930-checkpoint20-native-20261014.json.
The snapshot filename is historical; the verified payload contains25 updates.

Step51 passes optimized-source ac312d6 actual runner smoke:16x300, six updates,
interrupted/resumed, all three independent numerical gates pass in4.602s.52
source/software/protocol files are frozen. Independent log-duration-only E-step
versus scaled joint statistics on16x300 averages0.137762s versus0.066650s (2.067x)
in a local three-repeat arithmetic benchmark, not production/neural speed evidence.
23 focused tests/Ruff/Black/compilation pass. Statistical fits may now proceed
under the unchanged1930/1931/1932 three-arm contract; Oracle OFF. Raw smoke:
validation/results/joint-runner-scaled-smoke-native-20261014.json.

Step50 completes the actual checkpointed runner smoke at frozen source bbfc8df:
16x300 observations, two updates per arm (six actual), interrupted/resumed,
all independent terminal/outer/public numerical checks pass in6.024s. This is
explicitly reduced execution-only evidence; the statistical protocol is unchanged.
Before any full statistical fit, the joint backend replaces log backward sums
with the algebraically equivalent evidence-scaled probability recursion. All23
focused tests pass again in2.92s; Ruff/Black pass. Independent log E-step and
native autograd/filter/segment checks retain strict tolerances. Oracle OFF.
The optimized source must pass its own bounded smoke before frozen270-update fits.

Step49 implements the frozen-protocol three-arm joint Gaussian/duration runner,
atomic0/every5/final state/history saves, guarded exact resume and byte-preserving
completed no-op. Independent terminal/outer LL, learned-parameter autograd and
causal filtering/segment recursion verification are included. All23 focused tests
pass in3.42s; Ruff passes. Ten runner cases cover exact resumed states, frozen
emissions, actual public replay, corrupt/resealed checkpoints, headers and budgets.
Protocol unchanged; Oracle OFF. Source is saved before bounded actual runner
smoke, which must pass before statistical fits1930/1931/1932 (270 total updates).

Step48 completes the frozen-source joint EM execution smoke:16x300 observations,
three actual updates in0.782s, augmented objective -12625.720454 to -10379.349808
with no decrease. Source666013d, both source SHA256s and regenerated training
hash/counters independently verified. Initialization uses observations only;
Oracle OFF, no outer evaluation. Evidence: validation/results/
joint-gaussian-duration-smoke-native-20261014.json. Step47's21 focused tests pass.
Next: implement the checkpointed three-arm statistical runner and independent
replay for the frozen protocol. Joint statistical recovery remains untested;
this scalar Gaussian smoke does not establish neural or Trading acceptance.

Step47 validates the research joint scalar-Gaussian/fixed-bin duration EM backend.
Soft posterior occupancy updates means and variances; censored duration counts
retain positive support on all ages1..260. Known transition/initial parameters
remain fixed. All21 focused tests pass (2.66s), including public likelihood,
duration/mean/raw-variance gradients within1e-7, causal filtering within1e-8,
and monotone augmented marginal EM updates. Oracle OFF; production unchanged.
The three-arm protocol for seeds1930/1931/1932 is preregistered, not executed:
known-emission control, frozen observation-initialized emissions, joint emissions.
Next: frozen-source16x300 execution smoke, then checkpointed statistical runner.
No joint statistical, neural, Trading or legacy NPZ acceptance is established.

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

Step44 implements the frozen matched optimizer recovery runner. Atomic0/every5/
final phase checkpoints preserve exact Adam logits, first/second moments and
step count; source/config/software/protocol/data/phase guards fail closed.
Shared fixed-bin EM warm3 initializes both27-update arms; outer/true-law and
training-label exposure are evaluated only after both fits.35 focused tests
pass in3.52s including actual mid5/7 Adam interruption/resume and all-phase
short smoke. Oracle OFF. Next: run all predeclared1922/1923/1924 fits sequentially
at this frozen runner source, independently verify and checkpoint/doc/push.

Step43 matched score benchmark completes: same4x300 data,uniform fixed-bin
initial PMF,temperature1,age prior1e-6,clip5,Adam lr0.01,three repetitions.
Median public Autograd1.057941s vs analytic posterior score0.049636s (21.31x).
Objective error1.818989e-12,gradient error4.463097e-14,Adam step error3.469447e-18.
This is a frozen-nuisance research backend measurement, not a production/neural
speed claim. Benchmark source22a0f4bf and raw outcomes remain immutable.
Next recovery comparison is predeclared in
validation/fixed-bin-optimizer-recovery-protocol.json: fresh1922/1923/1924,
1024x300 training,24 outer sequences,shared EM warm3 then27 per matched
fixed-bin EM/Adam arm. Exact Adam moment/step checkpoints are required before
running. Runner implementation and execution remain pending; Oracle OFF.

Step42 implements an exact posterior-score backend for research-only fixed-bin
duration learning with frozen nuisance parameters. Six new tests pass, including
native public censored likelihood gradients and a clipped Adam update versus
expected-count gradients; unequal final bins and curvature prior are checked.
No production/neural fast path is introduced. Next matched benchmark is frozen
at fresh401921,4x300 rows,sigma2,three repetitions,identical uniform33-bin
initialization,temperature1,prior1e-6,clip5,Adam lr0.01; no outer evaluation.
Oracle OFF. No new recovery study has started yet.

Step41 validates accumulated optimizer execution on16x300 observations with
batch4, source b2c96267, one actual Adam update. Wall time6.174489s,
whole-process Linux peak582652 KiB (568.996 MiB), terminal target/reference
LL error0; known frozen nuisance preserved, no outer evaluation, Oracle OFF.
All195 materialized focused cases pass in9.14s. The prior4x300 smoke used
581704 KiB; measured peak differs by948 KiB across these separate runs.
These are resource observations, not equal-condition speed benchmarks or
full1024-sequence extrapolations. Chunking bounds the autograd sequence batch;
full input/initialization and reference-evaluation storage still scale with N.
Next: predeclare sufficient-exposure matched duration optimizer comparison,
using the same family and target objective; account for measured runtime before
allocating long fits. Neural/joint recovery remains unaccepted.

Step40 adds opt-in whole-sequence gradient accumulation via positive
optimizer_batch_size. Sum likelihood gradients, global penalty once, single
clip/Adam update per full batch; fixed-target scoring also respects chunk size.
47 accumulation/optimizer/training/verifier focused tests pass. Equivalence
covers loss/gradients/updates, uneven and empty sequences, external and neural
context; deterministic tolerances are explicit. Default batch behavior remains.
Next resource smoke is fixed at fresh seed401919,16x300 observations, chunk4,
one Adam update with frozen known nuisance parameters and no outer evaluation.
Prior frozen experiment/raw outcomes remain untouched; Oracle OFF.

Step39: actual4x300 one-step optimizer resource smoke completes successfully
at source d2d2389a. Constant-temperature/current-initialization Adam preserves
known frozen nuisance parameters. Optimizer wall time1.795389s, whole-process
Linux peak RSS581704 KiB (568.07 MiB), target/public/reference LL discrepancy
9.094947017729282e-13. No outer evaluation, oracle OFF. This is one bounded
execution sample, not an estimate of1024-sequence peak memory or convergence.
Next: implement/test exact gradient accumulation over bounded sequence chunks
before preregistering a sufficient-exposure duration-only optimizer comparison.
Keep one optimizer step/clipping operation per full accumulated batch and score
all restarts against the same fixed objective. Do not free neural/nuisance
parameters until that comparison passes its separately declared gates.

Step38 completes independent verification of the optimizer bridge pilot.
All nine terminal models pass training/outer/public LL, gradient and filtering
replays;21 source identities and regenerated data hashes match,18 actual pilot
updates,0 verification updates, completed resume is byte-preserving. Statistical
recovery is0/3 for each arm. These2-update runs are execution diagnostics only.
Ground-truth post-fit audit: no completed bull/bear training episodes in any
seed; true P(duration>60) is0.983017/0.773030 for bull/bear. A60-row shortcut
cannot substitute for the prior300-row larger-exposure recovery study.
Next: measure memory/time on a300-row actual optimizer smoke, then predeclare
a longer sufficient-exposure duration-only optimizer study before freeing
nuisance/neural parameters. Preserve all current outcomes; do not tune this
pilot after evaluation. Oracle remains OFF.

Step37 predeclares an execution-only optimizer bridge pilot on fresh
seeds1915/1916/1917:4 training and2 outer sequences of60 rows,2 updates each
for constant-temperature Adam, annealed Adam and unrestricted EM. All arms use
uniform full duration support and known frozen nuisance parameters. This is
not equal-compute, convergence or statistical recovery evidence. No arm is
selected. Source/software/protocol identities and completed-seed checkpoints
are guarded; labels/true PMFs are evaluated only after fitting. Syntax/static
and an actual2x20-row one-update smoke pass. Oracle OFF.

Step36: optimizer controls and fixed-target monitoring/restart scoring
implemented. Default update annealing and fresh restarts remain; constant
temperature and current-state initialization are explicit opt-ins. Evaluation
mode scoring removes dropout noise; histories distinguish target vs scheduled
pre-update objectives and final restart scores. All179 focused regressions pass
in12.30s. Archived study inputs are untouched and require an explicit historical
source manifest audit after Core changes. Next: frozen bounded bridge pilot,
not a large statistical recovery/joint-neural/Trading acceptance run. Oracle OFF.

Step35: bounded public-optimizer temperature diagnostic completed without
fitting or parameter mutation. At temperature1, `_compute_loss` matches public
likelihood exactly; default scheduling changes LL at fixed parameters
(span0.0046303224295911605 on2x40 observations). This proves a target-objective
distinction, not that annealing causes recovery failure. See
validation/optimizer-bridge.md for evidence and staged optimizer-control,
restart-score and parameter-unfreezing boundary. Production defaults and
completed frozen study remain unchanged; no new long fit started. Oracle OFF.

Step34: reusable read-only evidence auditor added as
`scripts.audit_block_mass_smoothing`. All three completed seeds and the batch
pass SHA, frozen source/software/contract, phase/grid/history/solver certificate,
shared-warm, recorded replay, numerical gate and summary arithmetic checks.
79 focused tests pass (14 new audit/tamper tests). This audit verifies recorded
independent replay evidence; it does not rerun numerical likelihood/E-steps.
Frozen study source140ff55 and all completed outcomes remain unchanged.
Next: isolate public optimizer behavior before unfreezing nuisance parameters;
see validation/optimizer-bridge.md. Oracle remains OFF.

Step33 completes the frozen1024-sequence fixed-bin mass-smoothing study on
fresh seeds1912/1913/1914. Each uses shared warm3 plus27 updates per arm;
171 total actual M-steps. All19 source/software/contract/data/phase/PMF/grid/
solver certificate/objective checks and independent terminal/outer/public
numerical/noop replays pass. Full gates: unsmoothed 3/3;
smoothed 3/3. Smoothing wins matched outer LL on3/3 seeds.
All raw reports are SHA-bound. Oracle OFF, source140ff55 and all predeclared
grids/strengths/budgets/gates unchanged. No production/joint-neural/Trading or
legacy NPZ acceptance is implied; the previous256 adaptive pilot is not a
paired sample-size control. Evidence: results/block-mass-smoothing-batch-native-20261014.json.

Next: review these matched outcomes before predeclaring any further estimator
or moving beyond known-nuisance duration-only validation. Preserve all current
results as completed evidence; do not adjust this study after outer outcomes.

Step32-1914-complete

Smoothing experiment seed1914 completes57 actual M-steps: warm3 and27
per matched fixed-grid arm. All19 source/software/contract/training/outer
identities, phase/budget/solver certificate/objective/PMF/grid checks, independent
terminal E-step and outer/public likelihood/gradient/filtering replays and
byte-preserving completed no-op pass. Both arms share initial PMF, grid and
observations. Smoothed minus unsmoothed outer LL0.9852910083482129 nats.
Main report SHA256d5627e3d41f6ae8f8cbc4cb9bd568f200651895ad5be9d6822019b8a02bb32ed. Oracle OFF, source140ff55 and
all declared widths/strengths/budgets/gates unchanged. No production/joint-neural
or legacy NPZ acceptance follows.
unsmoothed: TVs[0.06919848404346655, 0.07300840871324542, 0.08506009178729296, 0.07246084814552528], gain0.0077787162453354436 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].
smoothed: TVs[0.050312707755095235, 0.05431712146335517, 0.06936374964602543, 0.06923090398431361], gain0.007915562218717141 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].

Step32-1914-checkpoint-smoothed-15

Smoothing experiment seed1914 checkpoint: smoothed at15 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1914-checkpoint-smoothed-0

Smoothing experiment seed1914 checkpoint: smoothed at0 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1914-checkpoint-unsmoothed-0

Smoothing experiment seed1914 checkpoint: unsmoothed at0 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1913-complete

Smoothing experiment seed1913 completes57 actual M-steps: warm3 and27
per matched fixed-grid arm. All19 source/software/contract/training/outer
identities, phase/budget/solver certificate/objective/PMF/grid checks, independent
terminal E-step and outer/public likelihood/gradient/filtering replays and
byte-preserving completed no-op pass. Both arms share initial PMF, grid and
observations. Smoothed minus unsmoothed outer LL2.770644116533731 nats.
Main report SHA256f68c9fd8f4da6d3155a9b39ec7052eaa6b62cbc33f7d56b6beb7f49946946977. Oracle OFF, source140ff55 and
all declared widths/strengths/budgets/gates unchanged. No production/joint-neural
or legacy NPZ acceptance follows.
unsmoothed: TVs[0.08397400323627008, 0.07787619832170425, 0.09948752522398446, 0.07622428975784042], gain0.006373454804447647 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].
smoothed: TVs[0.06500096019116289, 0.06580853715708848, 0.0809680554664712, 0.07777723002743443], gain0.006758266487299554 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].

Step32-1913-checkpoint-smoothed-15

Smoothing experiment seed1913 checkpoint: smoothed at15 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1913-checkpoint-unsmoothed-27

Smoothing experiment seed1913 checkpoint: unsmoothed at27 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1912-complete

Smoothing experiment seed1912 completes57 actual M-steps: warm3 and27
per matched fixed-grid arm. All19 source/software/contract/training/outer
identities, phase/budget/solver certificate/objective/PMF/grid checks, independent
terminal E-step and outer/public likelihood/gradient/filtering replays and
byte-preserving completed no-op pass. Both arms share initial PMF, grid and
observations. Smoothed minus unsmoothed outer LL0.2281040401685459 nats.
Main report SHA256bb78904407ad2421854d484052a471daae62cb6ef3e17fc0706dfad0a87410d0. Oracle OFF, source140ff55 and
all declared widths/strengths/budgets/gates unchanged. No production/joint-neural
or legacy NPZ acceptance follows.
unsmoothed: TVs[0.07355676669601013, 0.08641623132355288, 0.09008035408903947, 0.06766988698701032], gain0.007734397346692074 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].
smoothed: TVs[0.04234832628750705, 0.06414683287189094, 0.07021739228445997, 0.07326639128710745], gain0.0077660784633821495 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].

Step32-1912-checkpoint-smoothed-15

Smoothing experiment seed1912 checkpoint: smoothed at15 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1912-checkpoint-smoothed

Smoothing experiment seed1912 checkpoint: smoothed at0 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step32-1912-checkpoint-unsmoothed

Smoothing experiment seed1912 checkpoint: unsmoothed at0 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

Step31 implements the checkpointed matched fixed-bin smoothing runner. All65
focused cases pass in13.56s (7 new runner cases plus58 existing). Actual
interrupted/resumed smoothed fits reproduce uninterrupted PMFs exactly;
phase/source/contract/data/solver/budget guards, shared warm/grid semantics,
post-fit-only public evaluation and byte-preserving completed no-op pass.
New2-file Ruff/Black/syntax and representative real short CLI/resume pass.
Next freeze this runner's source commit and execute all fresh seeds1912/1913/1914,
57 actual M-steps each. Source/runtime/contract/data are guarded; oracle OFF.

Step30 implements the separate fixed-bin block-mass smoothing M-step and
penalized marginal objective.14 new cases plus44 existing focused cases pass
(58 total,8.44s). Analytic unpenalized masses, spacing-aware curvature,
independent derivatives/Hessian, concavity, numerical stationarity, failed
solver rejection and actual short censored EM/public numerical checks pass.
Hessian-whitened native L-BFGS with up to3 budgeted Newton corrections meets
the unchanged1e-8 normalized-gradient gate. All new2-file static checks pass.
Evidence: docs/validation/results/block-mass-smoothing-preflight-native-20261014.json.
The1024-sequence experiment has not started. Next implement and verify its
checkpointed fixed-grid matched runner/resume before freezing execution source.
Existing production/historical experiment owners are unchanged; oracle OFF.

Step29 restores the native sparse execution snapshot against a4ace3f. All53
material files match connector Git blob identities after one evidence-backed
dependency reopen;44 existing focused tests pass in9.07s at unchanged
Python3.12.14/NumPy2.5.4/Torch2.14.1+cpu. No dependency installation is needed.
A fresh fixed-bin block-mass smoothing protocol is frozen before execution:
seeds1912/1913/1914,1024 training sequences, lambda0 versus log1024, shared
fixed-grid warm3 plus27 updates per arm. Solver, short-EM integration and
checkpointed runner verification precede any long fit. Oracle OFF.
See docs/validation/block-mass-smoothing-protocol.json and block-mass-smoothing.md.

Step28 completes all predeclared width-cap pilot seeds1909/1910/1911.
Matched256-sequence arms use warm3 and27 M-steps each;171 total M-steps.
All3 source/software/contract/data/PMF/phase/history/objective and independent
terminal/outer/public numerical/completed-noop verification checks pass.
Full recovery gates: unrestricted 0/3; width8 0/3.
Exact raw reports are SHA-bound by individual and batch verification. The
pilot changes the allowed partition class without true-PMF/label input; oracle
OFF. No1024-sequence, joint-neural/production or legacy NPZ acceptance follows.
Evidence: results/adaptive-width-cap-batch-native-20261013.json.
Width8 beats the control on outer LL for1/3 seeds. Every width8 state has a normalized feasible true-PMF projection with TV below0.12, while every fitted width8 state exceeds0.12. Thus this pilot removes the demonstrated partition obstruction but leaves a mass-estimation component in every width8 state; it is not a robust predictive improvement. This is finite-sample evidence, not proof that a particular smoother or more data will pass.

Next: predeclare block-mass smoothing/regularization with a matched larger
exposure study. Preserve all current widths, budgets and gates as completed evidence.

Step27-1911-complete

Width-cap pilot seed1911 completes57 actual M-steps (warm3 plus27 per arm).
All16 source/software/contract/training/outer identities, shared initial PMF,
matched budgets, phase/history/objective/positive full-age PMF/partition checks,
independent terminal E-step replays, regenerated outer evaluation, public
likelihood/gradient/filtering and byte-preserving completed no-op pass.
Width8 minus unrestricted outer LL: -41.36282740908973 nats.
Main report SHA256c6143fbf3d60243ed7a647260f0c54425034335206cce74fc85b1801b9faa08e. All results are pilot evidence,
not1024-sequence or production/joint-neural/legacy NPZ acceptance. Oracle OFF.
unrestricted: blocks[3, 4, 5, 4], TVs[0.17231219803297143, 0.11355436047328024, 0.1755661830409554, 0.1542870439059833], gain0.007597737959804615 nats/row, full gateFAIL, partition TV lower bounds[0.1659261942818482, 0.11049688153007527, 0.16907706199625655, 0.14704087707064722].
width8: blocks[33, 33, 33, 33], TVs[0.165242131703969, 0.17840184113484747, 0.16190821052544285, 0.12391747696442625], gain0.0018529008196532637 nats/row, full gateFAIL, partition TV lower bounds[0.02386179381417981, 0.024971122106164048, 0.051052002730845564, 0.04834243315574044].

Step27-1911-active-width8

Width-cap pilot seed1911 checkpoint: width8 at5/27,
blocks[33, 33, 33, 33]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

Step27-1911-active-unrestricted

Width-cap pilot seed1911 checkpoint: unrestricted at5/27,
blocks[3, 4, 5, 4]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

Step27-1910-complete

Width-cap pilot seed1910 completes57 actual M-steps (warm3 plus27 per arm).
All16 source/software/contract/training/outer identities, shared initial PMF,
matched budgets, phase/history/objective/positive full-age PMF/partition checks,
independent terminal E-step replays, regenerated outer evaluation, public
likelihood/gradient/filtering and byte-preserving completed no-op pass.
Width8 minus unrestricted outer LL: -28.033464356243712 nats.
Main report SHA25636a9e6e24cf2188fd5ced16008bf18f057a8b367d5747efd57178636281a6e57. All results are pilot evidence,
not1024-sequence or production/joint-neural/legacy NPZ acceptance. Oracle OFF.
unrestricted: blocks[4, 4, 5, 4], TVs[0.1512682124673306, 0.11415891768091632, 0.19243455817006525, 0.14708438297226797], gain0.005200034136191284 nats/row, full gateFAIL, partition TV lower bounds[0.13949747652704678, 0.11036037126376683, 0.18967903496131433, 0.1358535170371385].
width8: blocks[33, 33, 34, 33], TVs[0.16021754190356455, 0.22176273932976012, 0.21373524132051733, 0.12752437326266428], gain0.0013064974200463237 nats/row, full gateFAIL, partition TV lower bounds[0.023814457857965984, 0.02430026697124297, 0.04952418313143421, 0.05010040354108249].

Step27-1910-active-width8

Width-cap pilot seed1910 checkpoint: width8 at5/27,
blocks[33, 33, 34, 33]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

Step27-1910-active-unrestricted

Width-cap pilot seed1910 checkpoint: unrestricted at15/27,
blocks[4, 4, 5, 4]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

Step27-1909-complete

Width-cap pilot seed1909 completes57 actual M-steps (warm3 plus27 per arm).
All16 source/software/contract/training/outer identities, shared initial PMF,
matched budgets, phase/history/objective/positive full-age PMF/partition checks,
independent terminal E-step replays, regenerated outer evaluation, public
likelihood/gradient/filtering and byte-preserving completed no-op pass.
Width8 minus unrestricted outer LL: 15.788184175442439 nats.
Main report SHA2564d37e4ba58d861d44c6e6ee18d104e97159da263561ccd61ccae7f17d94cf5bc. All results are pilot evidence,
not1024-sequence or production/joint-neural/legacy NPZ acceptance. Oracle OFF.
unrestricted: blocks[4, 3, 5, 4], TVs[0.1310162390517595, 0.14796260880209172, 0.17532581976170813, 0.17586775837164556], gain0.005149776424069488 nats/row, full gateFAIL, partition TV lower bounds[0.12470794605793709, 0.14701677785676828, 0.17078985610860292, 0.17272074620117248].
width8: blocks[33, 33, 33, 33], TVs[0.185082841536105, 0.15511084872243294, 0.1490903352546657, 0.14827088207990077], gain0.007342579781769827 nats/row, full gateFAIL, partition TV lower bounds[0.02390236738745256, 0.02409627774125917, 0.05118498048109232, 0.05096372213525549].

Step27-1909-active-width8

Width-cap pilot seed1909 checkpoint: width8 at10/27,
blocks[33, 33, 33, 33]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

Step27-1909-active

Width-cap pilot seed1909 checkpoint: unrestricted at10/27,
blocks[4, 3, 5, 4]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

Step26 implements the separate research-only width-cap M-step/fitter and checkpointed paired driver. Exhaustive bounded partition optimality, inactive-cap equivalence, positive zero-exposure PMF, probability-preserving warm refinement, real censored short EM/interrupted resume, guarded source/contract/seed/prefix/history, post-fit public numerics and completed no-op checks pass.44 focused tests pass in12.94s; new3-file Ruff/Black/syntax checks pass. Existing adaptive/production owners and prior frozen source hashes remain unchanged. Next execute frozen256-sequence paired pilot, concurrency1, oracle OFF.

Step25 freezes a new width8 mechanism pilot before implementation/execution. Fresh seeds1909/1910/1911, matched256 training and24 outer sequences, warm3 plus27 updates per arm, fixed factor2, unrestricted versus width8,171 total M-steps, concurrency1. Full support1..260, oracle OFF and recovery/numerical gates unchanged. No label/truth enters fitting. This is not1024-sequence or production acceptance. See docs/validation/adaptive-width-cap-protocol.json and adaptive-width-cap.md.

Step24-batch completes the predeclared large-selection seeds1906/1907/1908.
All choose factor2 from inner validation. All3 numerical/integrity/replay checks
pass; full duration-PMF plus positive-outer-gain gates pass for0/3 seeds.
Seeds1907/1908 add60 full-data M-steps in this batch; each complete seed uses114,
342 total. Selected and frozen-factor2 controls reuse the identical fit per
protocol. Source/software/protocol match, training/outer datasets differ, and
all main reports are SHA-bound to independently verified final evidence.
Oracle OFF; support, factor grid, budgets and gates remain unchanged.

Next: use these post-fit diagnostics to design a new predeclared partition
experiment; do not retrospectively expand the completed protocol.

Step24-1908-complete

Seed1908 completes full_warm3 plus selected_refit27 on1024 sequences,
114 total actual M-steps. Selected factor2 and its matched frozen-factor control
reuse the identical fit. All13 source/software/contract/data/split/phase/history/
objective/PMF/partition checks, independent warm/refit terminal E-step replay,
regenerated outer evaluation and public likelihood/gradient/filtering checks
pass. Prior inner records and selection remain identical; completed CLI resume
is a byte-preserving no-op. Outer gain0.005763641079615051 nats/row;
PMF TVs[0.10355058387936868, 0.1102053532205235, 0.14247093861838375, 0.12593786647721306]; mean RMSE0.46875875257468874.
Full recovery gateFAIL; failed states['range', 'chaotic'].
Learned blocks[6, 4, 8, 5]; post-fit partition TV lower bounds[0.09669158388290576, 0.10879501011025602, 0.13445370129747347, 0.12447137080809408];
feasible forward-KL projection TVs[0.09928397853501024, 0.11083711265195112, 0.13702119954683634, 0.12789760322328086].
Verification is SHA-bound tob6cb1b8c5c04b4c00e255e25c5548bfcc09fb182106f2b4a4c5b12074ee92d79.
Post-fit diagnostics do not enter fitting/selection or change the gates. Oracle
remains OFF; no production/joint-neural/legacy NPZ acceptance is claimed.

Step24-1908-active

Seed1908 full_warm completes3 updates on1024 sequences; selected_refit
factor2 reaches5/27, blocks6/4/8/5.
All13 source/software/contract/data/split identities, prior inner phase records
and selected factor are preserved. Full-age PMF/partition/history/objective and
warm/refit continuity checks pass. This is recovery evidence; terminal replays,
outer evaluation and recovery/numerical gates remain pending. Oracle OFF.

Step24-1907-complete

Seed1907 completes full_warm3 plus selected_refit27 on1024 sequences,
114 total actual M-steps. Selected factor2 and its matched frozen-factor control
reuse the identical fit. All13 source/software/contract/data/split/phase/history/
objective/PMF/partition checks, independent warm/refit terminal E-step replay,
regenerated outer evaluation and public likelihood/gradient/filtering checks
pass. Prior inner records and selection remain identical; completed CLI resume
is a byte-preserving no-op. Outer gain0.00592318680781116 nats/row;
PMF TVs[0.10871667616130973, 0.11004879204839987, 0.12596853943506775, 0.11614778987526281]; mean RMSE0.8229498555275837.
Full recovery gateFAIL; failed states['range'].
Learned blocks[5, 4, 8, 6]; post-fit partition TV lower bounds[0.10845232583663918, 0.1092965235192142, 0.12497076145532046, 0.11436916048165213];
feasible forward-KL projection TVs[0.11094778037100295, 0.1146360225418479, 0.12722818351149318, 0.11685519102449432].
Verification is SHA-bound toe785101743f7cfb86ca146dbf3409969aef8e9ce5af97e8841e1ec2a6e0f2110.
Post-fit diagnostics do not enter fitting/selection or change the gates. Oracle
remains OFF; no production/joint-neural/legacy NPZ acceptance is claimed.

Step24-1907-active

Seed1907 full_warm completes3 updates on1024 sequences; selected_refit
factor2 reaches5/27, blocks5/4/8/6.
All13 source/software/contract/data/split identities, prior inner phase records
and selected factor are preserved. Full-age PMF/partition/history/objective and
warm/refit continuity checks pass. This is recovery evidence; terminal replays,
outer evaluation and recovery/numerical gates remain pending. Oracle OFF.

Step23-batch

Both seeds1907/1908 finish candidate_2 factor2:27 updates each,54 new
M-steps and168 cumulative including prior inner phases. All independent terminal
replays, regenerated datasets/splits, source/software/contract checks and prior
phase preservation pass. All three candidate validation scores and chosen factors
are independently reproduced. Selection is complete and saved before outer data;
outer evaluation remains null. Next full_warm (3 updates on1024 sequences) and
selected_refit (27), with frozen source3966159/method/budgets/gates and oracle OFF.
SHA-bound paired evidence: results/adaptive-large-selection-candidate2-batch-native-20261013.json.

Step23-1908-complete

Seed1908 candidate_2 factor2 completes27 updates (84 total M-steps).
Source/software/contract/data/split/history/objective/full-age PMF/partition and
independent terminal training E-step replay pass. All prior phases are preserved
exactly. All three candidates are independently scored on the same256 inner
validation sequences and the selection matches the maximum with stronger-factor
tie break. Factor2 validation LL-167164.1952762566, chosen factor2.0.
Blocks5/4/8/5. Main report SHA2568051085a684c3967fe17c83a9b303a9c7746e3af030623a56eaa2f30194f5f5d.
Selection is complete; outer evaluation remains null and full-data refit pending.
Frozen source3966159, method, budgets, gates and oracle OFF remain unchanged.

Step23-1907-complete

Seed1907 candidate_2 factor2 completes27 updates (84 total M-steps).
Source/software/contract/data/split/history/objective/full-age PMF/partition and
independent terminal training E-step replay pass. All prior phases are preserved
exactly. All three candidates are independently scored on the same256 inner
validation sequences and the selection matches the maximum with stronger-factor
tie break. Factor2 validation LL-166599.38091861576, chosen factor2.0.
Blocks5/4/8/5. Main report SHA256172a2ce05a5d12d8f8c7ee9cc269fc1f208c00ea9317619edfa5420c660f6f61.
Selection is complete; outer evaluation remains null and full-data refit pending.
Frozen source3966159, method, budgets, gates and oracle OFF remain unchanged.

Step23-1907-active

Seed1907 candidate_2 factor2 checkpoint: 5/27 updates,
blocks5/4/8/5. Source/software/contract/data/split/prior phase preservation,
history/objective/full-age PMF and partition checks pass. This snapshot is
recovery evidence; terminal independent replay and factor choice remain pending.
Native fitting continues at unchanged source3966159; oracle OFF.

Step22t completes candidate_1 factor0.5 for seeds1907/1908:27 updates each,
54 new M-steps and114 total including their prior warm/candidate_0 phases.
Both independent terminal E-step and validation replays pass; all13 source,
software/contract/data/split/history/objective/PMF/partition checks pass and prior
completed phase records remain identical. SHA-bound paired evidence is saved.
Selection and outer evaluation remain null. Next candidate_2 factor2 for both
seeds; frozen source3966159, method/budgets/gates and oracle OFF are unchanged.

Step22s persists seed1908 candidate_1 factor0.5 at5/27 updates,
blocks13/17/14/9. All13 source/software/contract and data/split identities,
preserved earlier fits, active budget/history/phase continuity, positive full-age
normalized PMFs and partitions pass. This active snapshot is recovery evidence,
not completed-candidate/outer acceptance.1907 candidate_1 is fully verified;
1908 fitting continues at unchanged3966159, oracle OFF.


Step22r completes seed1907 candidate_1 factor0.5:27 updates,57 total M-steps.
Prior warm/candidate_0 records remain identical. Source/software/contract/data/
split/PMF/partition/history/objective and independent terminal E-step checks
pass. Blocks8/14/10/9; inner validation LL-166628.70723221602, improved
by125.93604849430267 nats over0.125 on the same validation sequences.
Selection/outer evaluation remain null. Next seed1908 candidate_1; oracle OFF.


Step22q persists seed1907 candidate_1 factor0.5 at5/27 updates,
blocks8/14/10/9. Thirteen current source/contract/software identities, data/
split replay, preserved prior warm/candidate_0, active budget/history and positive
full-age normalized PMF/partition checks pass. This is an active checkpoint;
terminal replay/inner score and factor selection/outer recovery remain pending.
Seed1908 still awaits candidate_1; native1907 fitting continues at3966159.


Step22p restores the lost native environment. All54 current input/checkpoint
blob identities pass; declared CPU dependencies and editable install succeed
at matching Python3.12.14/NumPy2.5.4/Torch2.14.1+cpu. All47 focused cases and
44-file AST/Ruff/Black pass again. Prior fits/checkpoints are intact. Next
candidate_1 factor0.5 for1907/1908; no training outcome changed. Oracle OFF.


Step22o completes candidate_0 factor0.125 for both1907/1908:27 updates each,
54 new M-steps and60 total with their prior warm3. Source/software/contract/data/
split/PMF/partition/history/objective and independent terminal E-step checks pass;
both warm records are unchanged. Blocks1907:76/63/65/34;1908:68/80/63/37.
Inner validation LLs -166754.643281/-167330.094206. These are seed-specific
scores, not a cross-seed ranking. Both selection/evaluation/active fields remain
null. Next candidate_1 factor0.5 on each seed; source3966159 and oracle OFF stay
unchanged. Dataset/raw/verification/batch evidence is SHA-bound and pushed.


Step22n checkpoints seed1908 candidate_0 at10/27 updates, factor0.125,
blocks68/80/64/37. Source/software/contract and prior warm/data identity,
active budget/indices, positive normalized full-age PMFs, partitions and monotonic
fixed objectives pass. Terminal E-step/validation score remain pending while
native fitting continues. Seed1907 candidate_0 is complete and verified; factor
selection and outer evaluation remain pending for both. Oracle remains OFF.


Step22m completes seed1907 candidate_0, factor0.125, all27 updates at3966159.
Warm3 is unchanged. Thirteen source/software/contract identities, training and
768/256 split replay, full-age PMF/partition/history/objective checks and terminal
E-step reproduction pass. Blocks76/63/65/34; inner validation LL -166754.643281.
Total1907 work30 M-steps. Selection/evaluation remain null. Seed1908 retains its
verified warm3; next executes the same first candidate. Oracle remains OFF.


Step22l checkpoints seed1907 candidate_0 at10/27 updates, factor0.125,
blocks76/63/65/34. All13 source hashes and preserved prior warm/data/contract
identities, active budget/history, positive normalized full-age PMFs, contiguous
partitions and monotonic fixed objectives pass. Native fitting continues;
terminal E-step/inner score and selection/outer recovery remain pending. Seed1908
retains its verified warm3. Source3966159 and oracle OFF remain unchanged.


Step22k begins both remaining fresh large-selection seeds1907/1908: each
inner_warm completes3/3 updates at frozen3966159. Both bounded jobs exit0;
independent terminal E-step replay verifies all13 source/software/contract
identities,1024-data and disjoint768/256 split hashes, indices0..3, full positive
normalized PMFs/partitions and monotonic fixed objectives. Distinct data hashes
confirm separate generated datasets. Blocks1907:260/253/233/186;1908:254/254/
245/153. Selection/evaluation and active remain null. Next bounded resume starts
candidate_0 for each seed; no new recovery outcome is asserted. Admin global/
project/AGENTS layers are fully refreshed; active regime-v2 remains authoritative.
Oracle stays OFF; source/method/gates and the completed1906 result are unchanged.


Step22j completes seed1906's full large-selection experiment at3966159:
all114 M-steps, source/data/split/phase/objective/PMF replay and public numeric
gates pass. Inner validation chooses factor2, so selected/control reuse the
same1024 fit. Outer gain+0.00574134 nats/row; TVs0.095938/0.108081/0.143096/
0.126634 and mean RMSE1.448104. Full recovery fails range/chaotic. Their learned-
partition TV bounds0.125738/0.125731 already exceed0.12, so those partitions
cannot pass through block-mass updates alone. Larger training-only selection
does not resolve fine-PMF recovery in this seed. Known nuisance/oracle OFF
controls and all gates remain unchanged. Seeds1907/1908 remain to be executed;
no three-seed/public/joint Trading acceptance follows. Completed resume is a
no-op; raw and SHA-bound verification evidence are durably owned by validation.


Step22i persists seed1906's completed three-candidate selection: factor2 wins
on inner validation LL -166718.363173, ahead of factor0.5 by57.911853 nats.
The full1024 warm3 completes and selected_refit is at5/27,6/4/8/5 blocks.
Thirteen source hashes, immutable prior fits/data/contract, phase/objective/PMF
checks and warm-to-refit LL continuity pass. This is an active execution
checkpoint; final independent E-step/public numerical/recovery checks remain
pending. Selected/control fits are identical, so the completed fit will be
reused. Source3966159 and oracle OFF unchanged; outer evaluation remains null.


Step22h persists seed1906 candidate_2 (factor2) at update10/27,
blocks5/4/7/5. All13 source hashes, immutable prior completed phases/data/
contract identity, active budget/history, full positive normalized PMFs and
partition checks pass. No selection or outer evaluation yet; native fitting
continues to selection, full1024 refit and outer evaluation at unchanged3966159.
Oracle remains OFF. This is execution/recovery evidence, not final acceptance.


Step22g completes seed1906 candidate_1: factor0.5, all27 penalized updates
at unchanged3966159. Warm3 and candidate_0 remain exactly preserved. All13
source identities, training/inner split replay, PMF/partition/history/objective
checks and independent terminal E-step pass. Blocks18/13/19/9; inner validation
LL -166776.275026, better than factor0.125 by92.073033 nats on the same256
validation sequences. Completed work57 M-steps. Factor2 candidate is pending;
selection and outer evaluation remain null. Oracle remains OFF and retained.


Step22f persists seed1906 candidate_1 (factor0.5) at update10/27.
All13 source hashes and immutable prior warm/candidate_0/data/contract identity
pass; active budget/indices, normalized positive full-age PMFs, block coverage
and monotonic fixed history pass. Blocks18/13/19/9. This is an active
recovery snapshot; terminal E-step/inner score and selection remain pending.
Source3966159 and oracle OFF unchanged; native fitting continues.


Step22e completes seed1906 candidate_0: factor0.125, all27 penalized updates
at frozen3966159. Warm3 remains byte-for-byte identical. Thirteen source hashes,
training/inner split replay, full-age PMF/partition/history/objective checks and
independent terminal E-step replay pass. Blocks67/73/57/36; inner validation
LL -166868.348059. Total completed work30 M-steps. Selection and outer evaluation
remain null; no factor is chosen and no recovery is claimed. Next candidate_1
uses factor0.5 under the same source/contract. Oracle remains OFF and retained.


Step22d persists seed1906 candidate_0 (factor0.125) at update10/27,
with67/73/57/36 blocks. All13 current source hashes, prior immutable warm/data
identity, active budget/indices, positive normalized full-support PMFs and
partition consistency/monotonic history pass. This is an active execution
snapshot; independent terminal E-step and inner candidate scoring are pending.
Selection/outer evaluation remain null. Source3966159 and oracle OFF unchanged.


Step22c completes seed1906's inner warm3 on768 observations at source3966159.
All13 source/contract identities,1024 training and768/256 split replay hashes,
phase indices0..3, positive normalized full-support PMFs/partitions and fixed
objective monotonicity pass. Independent terminal E-step replay reproduces the
saved LL/objective without another M-step. Blocks254/250/228/129; selection and
outer evaluation remain null. Durable raw checkpoint is in validation/results.
Next resume starts candidate_0 (factor0.125), preserving the completed warm fit;
seeds1907/1908 and all large recovery gates remain pending. Oracle remains OFF.


Step22b implements the resumable large-selection driver and phase budget CLI.
Source/contract/software/data hashes bind resume; completed phases are re-scored
without new M-steps. Training-only selection is saved before outer evaluation;
factor2 control reuse and separate control fitting are verified. Fifty current
files are Git-blob restored (one test-helper scope reopen), declared CPU deps
and editable install succeed, and44-file AST/Ruff/Black plus the focused suite
and real short-fit/resume/public numerical smoke pass. Native environment is
Python3.12.14/NumPy2.5.4/Torch2.14.1+cpu. Large fits remain pending at this code
checkpoint; oracle remains OFF. Execution checkpoints belong to validation.md.


Step22a audits the existing small-inner selection and freezes the next
large-exposure experiment in docs/validation/adaptive-large-selection.md and
its machine-readable protocol. Fresh seeds1906/1907/1908 use768/256 inner
training/validation sequences, the unchanged three factors and warm3+fit27,
then full1024 selected/factor2 matched refits. Actual budget114..141 M-steps
per seed; no new results yet. env21 is lost and system Python lacks Torch.
Driver implementation, runtime restoration and static/real-fit smoke remain
before any long execution. Oracle support remains OFF and retained.


Step21e completes both fresh1024-only replications at frozen source8fcb7ca.
All60 new updates, source/data/phase/PMF/objective/metric and numerical gates
pass. Combined with1903, all90 large-arm updates are verified, but full recovery
passes only1/3 seeds:1903 PASS,1904 fails bearTV0.128586,1905 fails rangeTV0.125884
and chaoticTV0.130775. All three held-out gains remain positive.
Post-fit learned-partition bounds exceed0.12 for1904 bear(0.126560) and1905
chaotic(0.126638), proving insufficient resolution in those fitted partitions.
1905 range has a feasible projection below0.12 but its fitted masses miss the
gate, so representation bias alone does not explain every failure. The frozen
rule's recovery is not robust enough for public integration. Next: training-only
capacity/penalty selection at adequate exposure, with fresh outer evaluation.

Runtime loss was recovered through53 Git-blob-verified connector inputs and a
restored matching Python/NumPy/CPU-Torch environment with editable install.
All40 focused tests and AST/Ruff/Black over42 scoped Python files pass. Phase
checkpoints, guarded resume, completed no-ops and remote writeback are verified.
Factor2, prior1e-6, warm3+fit27, full ages1..260 and all gates stayed unchanged.
Oracle support remains off with its controls/code preserved. Known nuisance
parameters remain fixed; production/joint Trading/historical NPZ acceptance
is unchanged. Full raw/diagnostic/combined evidence is owned by validation.md.

Step20d completes paired128/1024 adaptive warm-start fits on fresh seed1903:
all60 M-steps and numerical/replay/objective gates pass. Fixed factor2 and
lambda2*log(count), warm3 plus penalized27 updates per arm, full ages1..260
and known nuisance parameters remain unchanged. Oracle support is disabled.
The1024 arm passes full duration recovery: TV0.11830/0.09873/0.10548/0.08922,
mean RMSE1.38373 and held-out gain+0.00792613 nats/row. The128 arm fails:
TV0.20355/0.16675/0.19051/0.15556, mean RMSE4.15282, gain+0.00752151.
Learned blocks grow from3/3/5/4 to5/5/9/7. Evaluation-only median-TV bounds
prove every small-arm partition exceeds0.12, while each large-arm partition
admits a forward-KL projection below0.12. This is one controlled paired seed;
the near-threshold bull result needs fresh-seed replication before integration.

Atomic phase/arm checkpoints and guarded resume are verified. Completed arms
are skipped, and restored phase evidence is re-scored before continuation.
38 focused tests, syntax/resume smoke and Ruff/Black pass; source stayed fixed
at421ef28 throughout the completed experiment. Production/joint Trading and
historical NPZ acceptance remain unchanged. The subsequent Step21 replication
above preserves this single successful paired result and the failed fresh seeds.

Step19a prepares a fresh paired warm-start/penalty pilot (seed1902). Factors
0.125/0.5/2 are compared from uniform and observation-only three-update EM
starts, with equal nine-update inner paths and twelve-update outer paths.
The48/16 inner split,64 training/16 held-out sequences of300 rows and all84
actual M-steps are fixed before evaluation. Shared warm-up is computed once;
each refit warm-up uses its own training observations. Initial PMFs/partitions
are validated and copied. No oracle support or production changes. Results
complete in Step19b; Step18's negative pilot is preserved at its immutable source
commit. All35 focused tests, AST/warm smoke and Ruff/Black checks pass. Both
initializations select factor2. Warm-start TV improves bull/range/chaotic versus
uniform, with bear unchanged at0.40826 (one uniform block); all three fits,
including fixed-width16, fail full recovery. Warm mean RMSE16.54 remains worse
than the fixed control5.30. All84 updates, replay/source hashes, warm continuity,
PMF/objective/metric gates and public numerical checks are verified. Next
research boundary is sample-size replication before further penalty tuning;
the weaker inner candidates had substantially worse validation likelihood.

Step18a adds a research-only adaptive full-support duration histogram. Exact
dynamic programming learns contiguous block boundaries from posterior duration
counts; a symmetric1e-6 pseudocount preserves every configured age. The EM
objective includes a fixed penalty per extra block and is checked for monotonicity.
Penalty selection sees training observations only. Nine new tests pass, including
exhaustive partition comparisons, learned boundaries, public gradient/recursion
checks and split isolation. Together with24 unchanged focused cases,33 distinct
cases are verified. Syntax, smoke, Ruff and Black pass for both added files.
The predeclared pilot uses fresh seed1901,64 training/16 held-out sequences of
300 rows,48/16 inner selection,three six-update candidates and two ten-update
refits (adaptive and fixed-width16 control):38 M-steps total.
Step18b completes and verifies this pilot. Likelihood selection chooses factor2;
adaptive blocks are3/1/1/2 by state. Both arms are held-out-positive but fail
recovery. Adaptive TVs0.17578/0.40826/0.46164/0.29185 and mean RMSE22.24 are
worse than the fixed control (RMSE5.69, TVs0.17650/0.15673/0.20728/0.22982).
Bear/range stay uniform. All numerical/replay/objective checks pass. This pilot
does not repair recovery; penalty/initialization need a fresh-data comparison.
No production/default/oracle-mode changes or three-seed recovery claim.

Step17a adds guarded --resume and --stop-after-seed to the full-support batch.
Completed seeds are retained only when source/input/runtime/contract match and
all20/30-update histories are complete. The model, data and acceptance budgets
are unchanged.24 focused tests cover mechanics plus bounded CLI resume/skip.
The initial full-support attempt was interrupted after the last reported
small901 update30, before any completed seed report; it supplies no acceptance.
Step17d completes all three full-support seeds,360 actual EM updates and15
monotonic fits. Every inner selection chooses width16. All six arms improve
held-out likelihood over their own full-support uniform baseline, but all six
fail the unchanged full recovery gate. Large-arm bull/bear/range TVs all pass;
chaotic TVs are0.14085/0.14014/0.15549 versus the0.12 gate. Source/input/replay
hashes, complete histories, compact PMFs and public numerical checks pass;
maximum gradient/LL/state-probability errors are2.05e-14/1.23e-11/8.45e-14.
An evaluation-only representation bound proves full-support width16 cannot
achieve chaotic TV below0.136925, independent of data volume or optimization.
The likelihood-selected family has insufficient shape resolution; this gate
failure is not evidence of a recursion defect. See validation.md for the
full report, paired oracle comparison and exact block-bound derivation.

Oracle duration support is now disabled by default in the compact research CLI:
--support-mode full uses ages1..260 for every state. --support-mode oracle
retains the per-state bounds as an explicit control; no oracle code or evidence
is removed. Training, public filter/likelihood, gradient checks and uniform
baseline all use the selected mode.22 focused tests pass. The full generic
three-seed recovery experiment is complete with failed recovery; the accepted Step15
statistics below remain oracle-support evidence. Known emission/transition/initial
parameters remain controlled in both modes. Width16 has72 parameters in full
mode, versus53 in oracle mode.

Step15 completes the compact-duration batch. All three inner validations select width16, reducing duration degrees of freedom from757 to53. With1024 observation-only sequences/seed, all three fits pass the unchanged full duration gate: each state PMF TV<=0.12 (observed0.05292–0.11169) and positive held-out gain0.00323–0.00379 nats/row. The128-sequence arms fail full recovery; seed903 also remains held-out-negative. Compactness improves every state/seed TV and held-out score versus the unrestricted Step14 fits, while sacrificing training likelihood. Details: [validation.md](validation.md#compact-duration-family-with-training-only-selection--2026-10-13).

Step15 verification:18 focused tests; AST/Ruff/Black pass over39 verified materialized/new Python files. All360 EM updates and15 monotonic fits, replay/source/input-report hashes, disjoint inner selection and compact support/density invariants are verified. All3120 checked production duration-gradient coordinates agree within3.01e-14; six public/reference likelihood errors<=1.14e-11 and state-probability errors<=8.66e-14. This accepts label-free duration recovery only under known support/nuisance parameters, finite D260 and valid starts; it is not stock joint optimize or Trading-feature acceptance. Production defaults remain unchanged; research masks/family have no public runtime/artifact contract. Historical NPZ verification remains open. Next research boundary: a support-independent family with enough shape resolution, selected from training observations and evaluated on fresh data, before public integration or joint nuisance/Trading learning. Earlier Steps12–14 remain separate evidence.

## Structural research findings — 2026-10-12

The causal emission self-conditioning defect is repaired; the former counterexample now integrates to 1.0. Noncausal internal generative likelihood/training is explicitly rejected. An exact self-reset/tail counterexample still disproves unrestricted episode-duration identifiability. The native fit has imperfect observable run laws despite perfect hard-state recovery. Historical findings: [validation.md](validation.md#structural-diagnosis-of-poor-recovery--2026-10-12). The controlled encoderless comparison is now reported below; duration identifiability and rare-tail precision remain bounded research questions.

## Latest code audit — 2026-10-12

The repository-wide code audit is complete: 787 CPU tests passed, three CUDA-only tests skipped, and AST/Ruff/Black/diff checks passed. Ten production defects were corrected and protected by eleven new regression tests. Legacy emission initializer behavior remains unchanged. Detailed scope, findings, diagnostic corrections and evidence are owned by [testing.md](testing.md#complete-code-audit--2026-10-12).

Earlier 30/40-iteration diagnostic descriptions refer to maximum budgets: the audit found that convergence_stop=False was ignored. Actual fixed-length comparisons and full post-fix statistical acceptance require renewed evidence. The code verification gate is green. The subsequent native OpenAI-workspace pilot below records actual completed optimizer budgets.

## Current status

**Phase:** independent HSMM correctness milestone complete; package core is accepted for the next integration/research gate. The current model core has been audited for distribution semantics, HSMM recursion, variable-length handling, runtime state, context routing, and Neural context modulation.

Current maintained defaults and contracts:

- ModelConfig.use_context_encoder=False; the internal ContextEncoder is opt-in.
- External context remains independently supported through context_dim.
- The canonical production causal path is the NHSMM causal filter/runtime contract documented in [model.md](model.md).
- n_states=3 remains the current Nautilus baseline from the maintained 3-vs-4 usability study.
- Historical machine-readable validation evidence is retained under [validation/README.md](validation/README.md).
- The full controlled package acceptance suite was re-run batchwise on 2026-10-06 after the ContextEncoder opt-in migration; duration, state recovery, learning/prediction, and transition-context gates all passed.

## Accepted package-core evidence

Controlled package evidence currently supports these narrow claims:

- **Duration context:** PASS for strong/moderate recovery with a bounded null control.
- **Advanced duration tail:** PASS on the maintained synthetic D+ gate; true-tail data improve survival calibration while the finite-support control drives `h_tail` to `1.0` and remains behaviorally equivalent.
- **Latent-state recovery:** PASS up to permutation when the controlled acceptance harness explicitly uses K-Means initialization.
- **Learning and held-out prediction:** PASS for the maintained strong/moderate/weak contracts; null remains a negative control.
- **Transition context:** PASS for strong/moderate recovery with near-zero null effect under the maintained refinement policy.
- **Multi-component identifiability:** historical PASS across the frozen controlled gate; not a general identifiability guarantee for unrestricted self-reset/tail models.
- **General-context robustness:** package-core PASS under split-fit direction replication.


A separate labeled synthetic trading-regime testbed is defined in [trading-regime-dataset.md](trading-regime-dataset.md) for controlled state/duration/transition usefulness experiments. It is not real-market evidence and does not replace Gates 11–14.

These claims are package/mechanism evidence only. They do not establish trading utility, market semantics, or downstream predictive value. See [validation.md](validation.md) for the maintained metrics, thresholds, and reproduction commands.

## Native continuation after audit — 2026-10-12

Seed 901 completed finite/tail diagnostics in the native OpenAI workspace, with observed 4/40 joint and 80 duration-only steps. Longer joint training reduces duration TV to 0.200/0.227 (finite/tail), while state accuracy remains 100%. Fixed temperature again worsens tail episode NLL. This one-seed diagnostic does not change defaults or close statistical acceptance. Evidence is owned by [validation.md](validation.md#model-conform-duration-pilot-after-audit-native-workspace--2026-10-12).

## Trading synthetic recovery — latest diagnostic

The 2026-10-12 model-conform duration-law pilot finds that all fits recover hard states exactly while their latent duration laws remain imperfect. Increasing the production joint-training maximum budget from 4 to 40 iterations reduces median duration total variation from 0.310/0.327 to 0.192/0.226 (finite/tail). Fixed temperature improves TV slightly but worsens median labeled tail episode NLL; no default change is justified. Duration-only learning with known frozen nuisance parameters at 80 iterations reaches TV 0.118/0.111, with a different budget. Joint self-transition probabilities remain high; one state's complete tail coverage is only 2/3/0 episodes across seeds. That pilot did not establish a mathematical defect. Its metrics predate the subsequent code audit above; Trading recovery and full post-fix statistical acceptance remain open. See [validation.md](validation.md#model-conform-duration-law-pilot--2026-10-12).

## Production hardening closed on 2026-10-05/06

### Distribution and runtime

Closed and regression-tested:

- distribution initialization no longer double-applies context;
- emission/K-Means initialization preserves registered parameter identity;
- runtime score caches invalidate after parameter replacement;
- decode normalizes list input before batch-shape access;
- log_likelihood rejects positive infinity rather than mapping it to a finite maximum;
- unknown distribution activation names fail closed.

### Mathematical and PyTorch correctness

Independent checks verified:

- causal state-age filtering against probability-space brute force;
- non-causal forward likelihood and Viterbi against complete small-model segment/state enumeration;
- batch vs single-sequence likelihood and decoding consistency;
- Gaussian and Student-t emission likelihoods against PyTorch references.

Closed defects:

- variable-length list likelihood/training padding leakage;
- padded-row NaN autograd in forward recursions;
- duplicate public external-context routing;
- max_duration=1 empty duration-regularizer NaN;
- scheduled distribution temperatures excluded from the standard optimizer.

### Model boundary and lifecycle

Closed and regression-tested:

- public inference requires initialized distributions with a defined runtime error;
- observation rank, feature width, and finiteness validate at the model boundary;
- empty-sequence likelihood behavior is consistent;
- structurally incompatible optimize(..., cfg=...) configs fail closed;
- optimization preserves the caller's train/eval mode across restart-created distributions and validation failures;
- model construction owns a resolved config copy, so explicit encoder injection cannot mutate a reusable caller config and leak Neural opt-in into later models;
- variable-length external context validates against `context_dim` independently of observation feature width.

### ContextEncoder opt-in

The internal learned observation-derived context path is optional:

- default: no ContextEncoder, no distribution context networks unless external context capacity is explicitly configured;
- use_context_encoder=True enables the canonical internal encoder;
- passing an explicit encoder is also an opt-in;
- runtime, filtering, diagnostics, artifacts, and tests support both paths;
- artifact v2 records context timing; v1 encoderless artifacts remain compatible, while v1 encoder artifacts require retraining and are rejected.

### Neural semantics

Closed and regression-tested:

- categorical log_prob rejects fractional, non-finite, and out-of-support labels instead of silently truncating them;
- emission context deltas remain state-centered while respecting max_delta;
- the unused internal Neural.alpha parameter was removed;
- temperature remains categorical-logit semantics and does not alter continuous emission location parameters;
- categorical temperature controls fail closed for non-finite/non-positive values and their stored parameters are frozen;
- context-free `delta_scale` controls are frozen because they are behaviorally inactive without context networks.

## Verification

Latest standalone core gate after the package audit fixes:

    ruff check nhsmm tests scripts: PASS
    black --check nhsmm tests scripts: PASS
    git diff --check: PASS
    focused causal-context regression suite: 37 passed
    python -m pytest -q: 712 passed, 3 skipped

The three skips are CUDA-only tests on the CPU workspace. A public-API artifact/inference/runtime round-trip also passed: artifact save/load preserved batch likelihood exactly, artifact-loaded and `load_inference_model(...)` runtimes produced identical posteriors across 17 causal streaming steps, state/age posteriors remained normalized, reset semantics passed, inference parameters were frozen, and variable-length list likelihood remained finite.

Post-fix causal-context revalidation on 2026-10-07 also passed on `9c4429b`: duration context returned to `13/15` positive moderate seeds without threshold changes, and transition context retained strong/moderate median correlations of `0.999987`/`0.993766` with a null median absolute context delta of `6.90e-7`.

Full controlled revalidation on 2026-10-06 also passed:

- duration context: strong/moderate/null PASS;
- latent-state recovery: strong/moderate/null PASS;
- learning/prediction: strong/moderate/weak learning+usefulness PASS; null learning-control PASS with usefulness not applicable;
- transition context: strong/moderate/null PASS.

The validation harnesses were run in scenario-sized batches with persistent JSON artifacts under the AWA workspace work store. See [validation.md](validation.md) for the current measured summaries.

## External HSMM reference cross-check

A fixed-parameter cross-check against the independently developed `gaussian-hsmm 0.1.0` explicit-duration implementation was completed on 2026-10-07.

Scope:

- non-causal stationary explicit-duration HSMM semantics shared by both implementations;
- identical initial probabilities, zero-diagonal boundary transitions, explicit duration distributions, and diagonal-Gaussian emission log probabilities;
- 20 deterministic random small models with `K=3`, `Dmax=5`, `F=2`, `T=11`;
- comparison of total marginal log likelihood and global Viterbi/MAP state path;
- fixed parameters only; optimizer/training behavior was intentionally excluded.

Results:

- 20/20 Viterbi paths matched exactly;
- maximum absolute log-likelihood difference: `2.84217094304e-14`;
- median absolute log-likelihood difference: `3.5527136788e-15`;
- Viterbi mismatches: `0`.

This is external implementation evidence for the shared classical non-causal explicit-duration core. It does not validate NHSMM-specific causal dynamic-hazard semantics, context modulation, Neural encoding, training policy, or downstream market usefulness.

## Nautilus usability boundary

The maintained 2026-10-05 3-vs-4 state-count study favored three states:

- 3-state runs were healthier and had slightly better mean OOS log likelihood per step;
- four states added soft posterior capacity without a hard-state usage gain;
- operational baseline therefore remains n_states=3.

That study predates the current real-data integration gate and is not a substitute for a fresh current-feature replay.

## Research/correctness milestone — independently verified HSMM core

The next package-level research milestone is to raise the HSMM core from regression-tested correctness to independent mathematical verification. This milestone is package-internal and precedes any renewed downstream Nautilus acceptance claim.

Execution is staged by [hsmm-correctness-work-gates.md](hsmm-correctness-work-gates.md).
Gate 0 mathematical semantics are frozen in [hsmm-mathematical-contract.md](hsmm-mathematical-contract.md). The audited non-causal duration-dependent Viterbi predecessor-collapse defect is fixed and regression-tested. Gates 1–10 are complete: independent exhaustive reference, production likelihood equivalence, causal filtering/posterior equivalence, Viterbi/MAP equivalence, structural invariants, gradient correctness, deterministic property-based state-space exploration, measured numerical precision bounds, statistical recovery revalidation, and final package acceptance.

The milestone is complete only when all of the following are observed:

- an intentionally simple exhaustive reference implementation enumerates all valid segment/state paths for small `K`, `D`, and `T` without importing production recursion, filtering, hazard, or runtime helpers;
- production `log_likelihood`, causal filtering, state/age posteriors, episode-boundary probabilities, and Viterbi decoding agree with that independent reference across the maintained small-model matrix, including `K=1`, `D=1`, `T=1`, self-transitions, impossible durations, and near-deterministic probabilities;
- causal batch forward, stateless model filtering, and stateful runtime agree within an explicitly measured Float32 numerical contract, while a Float64 reference gate uses materially tighter tolerances;
- end-to-end likelihood gradients for initial, duration, transition, and emission parameters pass `torch.autograd.gradcheck` or an equivalent finite-difference reference on small Float64 models;
- property-based tests cover normalization, causal prefix invariance, padding invariance, streaming equivalence, support preservation, and episode-age reset semantics;
- the full canonical pytest/static gate remains green after the new correctness harness is integrated.

Passing this milestone supports the precise claim: **the maintained HSMM recurrence and outputs are independently verified against exhaustive small-model reference calculations**. It does not imply formal proof for arbitrary floating-point executions, statistical identifiability on every dataset, or downstream trading utility.

Fixed-score core verification is separate from the predictive neural repair evidence documented above. Gate 10 closed on 2026-10-07 with Ruff, Black, diff-check, and the canonical CPU suite green (`668 passed, 3 skipped`).

Downstream integration is a separate post-milestone scope and is not evidence for this core correctness claim.

## Next milestone — real-market HSMM hypothesis tests

The next research block is defined in [market-hypothesis-gates.md](market-hypothesis-gates.md) as Gates 11–14:

- Gate 11: non-geometric regime duration structure;
- Gate 12: predictive state separation;
- Gate 13: incremental episode-age information beyond state alone;
- Gate 14: replicated OOS utility against HMM/state-only/no-regime baselines.

These gates are intentionally falsifiable and downstream-facing. They do not reopen Gates 0–10 and cannot weaken the accepted package-core correctness claim.


## Catalog prerequisite for Gates 11–14

Real-market hypothesis execution is currently **blocked pending completion of the canonical Nautilus catalog**. The package-core milestone remains complete and unaffected.

Before Gate 11 starts, the downstream data foundation must be completed and verified first. No temporary or partial catalog, simplified OHLC substitute, or ad-hoc feature reconstruction is accepted as evidence for Gates 11–14. The later market-hypothesis runs must consume the maintained Nautilus causal feature pipeline and its canonical catalog contract.
