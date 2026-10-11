# Stage 2V.9 — Full variance-aware recovery and benchmark audit

## Full recovery protocol

Source HEAD: `21b0eaad4f47510edd88ac38d6e4700ba78be5e3`, branch `regime-v2`.
Dataset: `data/reference/trading-regimes-v1/trading_regimes.npz`.
Seeds 901, 902, 903; three process workers; 24 training and 8 evaluation windows of 480 bars; 30 joint iterations; 12 transition-refinement steps for the context arm.
Only emission initialization changed from kmeans to kmeans_cluster_variance.
Ground-truth labels were used for evaluation, never for unsupervised training.
Python 3.14.7, Torch 2.9.1+cpu, NumPy 2.5.3.
The full job completed in 1476.222 seconds with exit code 1 because acceptance failed; stderr was empty.
Syntax/CLI and a reduced 2-iteration all-arm smoke passed before the full job.
Scripts were executed with PYTHONPATH=/workspace; the existing environment is not an editable package installation.

| Median metric | Previous kmeans | kmeans_cluster_variance | Threshold |
| --- | ---: | ---: | ---: |
| State accuracy | 0.490104 | 0.555208 | >= 0.65 |
| ARI | 0.173976 | 0.230769 | >= 0.40 |
| Boundary F1 +/-5 | 0.135693 | 0.124688 | >= 0.45 |
| State-run median relative error | 0.891551 | 0.890650 | <= 0.50 |
| Transition MAE | 0.236082 | 0.203757 | <= 0.20 |
| OOS likelihood per row | -8.830732 | -7.731145 | higher is better |
| Full minus no-context HSMM likelihood | -0.054567 | -0.036904 | >= -0.001 |

State recovery and likelihood improve, but overall acceptance remains false.
The metric defects below limit interpretation. The duration number measures filtered-argmax state-run fragmentation, not recovered duration-distribution parameters.
All four predicted states occur in every full-run seed, so missing-state exclusion did not affect this particular run.

## Confirmed benchmark/generator defects

### 1. Transition context target mismatch

In generate_trading_regime_dataset.generate, _next_state is called once before the episode's bar loop using macro_value at episode start. Its probability vector is copied to gt_p_next_* for every bar in the episode. In validate_trading_regime_usefulness._aligned_transition_mae, the model is evaluated with context at the episode-end bar.

The causal NHSMM boundary contract conditions on the last observed context before the next observation. The generator instead selects the next episode using start context. Therefore this test does not isolate recovery of an end-context transition law.

On the exact full-run eight test windows, 42 evaluated boundaries:
- start-context generator formula versus stored target MAE: 0.0;
- end-context generator formula versus stored target MAE: 0.022810575752340568;
- mean absolute macro-context start/end change: 0.37577276988003716.

This is a target-definition mismatch, not a correction to subtract from observed model MAE. The same end-context model could partially learn start information through other conditioning, so this measurement does not establish a strict achievable-error floor. Preserve v1 data; fix the evaluation contract or introduce a separately versioned end-context dataset before reinterpreting acceptance.

### 2. Boundary matching loses feasible matches

_boundary_f1 greedily chooses pairs by minimum absolute distance. This need not maximize the number of one-to-one matches within tolerance.

Reproduced counterexample: true boundaries [6,10], predicted boundaries [1,6], tolerance 5. Greedy selects 6->6 and reports F1=0.5. Matching 1->6 and 6->10 is valid and reports F1=1.0.

Use maximum-cardinality matching for tolerant boundary F1. Full-run state predictions were not persisted, so the size of this defect on the recorded full-run metrics has not been measured.

### 3. Missing states are excluded from duration error

_run_medians returns NaN for a state with no predicted runs. _one skips it when aggregating duration relative errors. A collapsed predictor can therefore receive a misleadingly favorable score on remaining states instead of failing state-duration coverage.

Reproduced run medians for prediction [[0,0,0,0]]: state 0 has 4.0; states 1,2,3 are missing. Explicit coverage failure or a declared missing-state penalty is required. Window-edge runs are also censored, so this operational metric must remain distinct from intrinsic episode-duration recovery.

### 4. Invalid generator annotations

typing.get_type_hints fails on _duration because np.RGEnerator does not exist, and on _next_state because np.RGenerator does not exist. Both should use np.random.Generator. The return annotation np.array denotes a function rather than the np.ndarray type. Postponed annotations hide these errors during ordinary generation; they do not explain numerical recovery results.

## Other interpretation limits

The earlier supervised Gaussian classifier and NHSMM emission-only pilot use different OOS samples and inference protocols. They do not establish an upper bound on achievable temporal recovery.
Mean-separation feature ratios do not measure variance/covariance or temporal discrimination.
No mathematical recursion defect was established by this audit. Production model code and global defaults remain unchanged.
Further model debugging must separate parameter-duration recovery, decoded state runs, causal filtering, and retrospective MAP decoding.

## Evidence

- results/full-recovery-kmeans-baseline-2026-10-11.json
- results/full-recovery-cluster-variance-2026-10-11.json
- results/emission-init-recovery-smoke-2026-10-11.json
- results/benchmark-audit-2026-10-11.json

These are recorded measurements, not passing acceptance.
