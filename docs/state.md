# NHSMM State

Current package readiness and next package-level boundary. Detailed model semantics are in [model.md](model.md), configuration and tuning in [configuration.md](configuration.md), validation semantics and accepted package evidence in [validation.md](validation.md), verification policy in [testing.md](testing.md), and release procedure in [release.md](release.md).

## Stepwise repair — current checkpoint

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
