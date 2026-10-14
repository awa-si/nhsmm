# Joint static-chain findings

The preregistered study completed 270 EM updates across seeds 1934, 1935,
and 1936. Joint learning of Gaussian means, variances, fixed-bin durations,
static transitions, and initial probabilities passes every recovery and
numerical gate in all three seeds. Oracle remains disabled.

| Arm | Full gates passed | Duration gates passed | Transition gates passed |
| --- | --- | --- | --- |
| Known-chain information reference | 3/3 | 3/3 | 3/3 |
| Estimated chain held frozen | 0/3 | 3/3 | 0/3 |
| Estimated chain learned jointly | 3/3 | 3/3 | 3/3 |

All nine fits also pass mean, variance, initial-probability, outer-gain,
and numerical gates. The deliberately incorrect frozen transition matrix
fails only the transition gate. These results do not support a claim that
incorrect static transitions destroyed duration recovery in this experiment.

The joint model's maximum duration TV across twelve fitted states is
0.105773 (gate 0.12); maximum transition-row TV is 0.035493 (gate 0.10);
initial TV is at most 0.022923 (gate 0.06). Maximum mean error is 0.029425
and variance relative error is 0.009592, below their 0.5 and 0.25 gates.

Joint learning improves outer log likelihood over the initialization-matched
frozen arm by 2.293196, 9.794568, and 1.118997 nats respectively. The gains
are positive in all seeds but modest. Each outer set contains 7,200 rows.
The known-chain arm is a separate information reference, not an
initialization-matched optimization comparison. Each arm's gain against
its own initial model includes emission and duration learning as well as
any chain learning.

The frozen experiment source is 2294c6e2ccd9ced0796ae640840c06f811e22601.
Each report binds 60 source files, software versions, the immutable protocol,
data hashes, initialization, all five parameter groups, and update history.
Independent verification replays terminal full-training and outer likelihoods,
public gradients for all five groups, segment likelihoods, and causal filters.
Completed resume is byte-preserving and adds no updates. Verification adds
zero fitting updates. The final batch binds raw reports and proofs by SHA256.

Seed 1936 was restored from its durable 85-update checkpoint after local
state loss. Only its remaining five EM updates were repeated; replay recovered
the same previously recorded terminal statistical and numerical results.
The recovered runner's 34 focused tests, Ruff, Black across 208 Python files,
and compilation passed before final writeback.

No labels or true-law duration masks enter learning or selection. All duration
ages 1 through 260 retain positive support. Truth alignment and completed versus
censored episode exposure are evaluated only after fitting. No seed, model,
initialization, budget, or gate was tuned after observing outer results.

This closes the bounded all-static study. The next boundary is neural context
and the production optimizer path. It establishes neither global convergence
nor neural, trading, or legacy binary-fixture acceptance. CPU suite coverage
and its explicit CUDA/legacy-fixture limits are recorded in the final validation
artifact and testing documentation.
