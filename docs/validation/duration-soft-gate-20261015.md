# Duration soft-gate API consistency — 2026-10-15

Audit D07 is resolved against baseline c36decbe32bfc558f2d505b95d4084086b746c9d on regime-v2.

Duration now applies soft_dmax in its shared modulation path. Given identical context, temperature, mask and gate, expected_probs, softmax(forward), forward(return_dist=True).probs, log_prob and log_matrix describe the same finite bucket PMF. Gate shapes remain [D] and [K,D]. Generic helpers retain their existing context/batch shapes; log_matrix retains its padded shape and T expansion.

The gate remains sigmoid(soft_dmax).clamp_min(EPS), multiplied into the masked, temperature-adjusted PMF and renormalized. Temperature does not rescale the gate. Hard masks stay exact zeros. The final D+ bucket and separate tail hazard retain their existing semantics; a categorical bucket distribution is not the complete unbounded episode distribution. State keys and artifact schema are unchanged. The model bias must still be explicitly supplied when using component helpers.

## Evidence

- A new regression test failed against the baseline because expected_probs ignored the gate.
- New public-API regression coverage: 33 cases including global/state gates, sequence/timestep/nested contexts, nonunit temperature, saturated gates, exact masks, parameter gradients, invalid gate shapes, T expansion and separate tail hazard.
- Distribution smoke: 118 passed.
- Expanded duration/context/filter/causal validation: 121 passed. These blocks overlap; counts are not additive.
- Native Nautilus/NHSMM/Qlib/GBM integration: 49 passed, 1 explicit skip for an isolated no-GBM profile, 22 existing NumPy/pandas deprecation warnings.
- Independent comparison to the baseline source: 40 float32/float64 cases have bitwise-identical log_matrix results and logits gradients.
- New test file full Ruff and distribution source Ruff F checks passed; git diff --check passed.
- Initial lint invocation could not create its cache in the read-only worktree; rerun with --no-cache passed.

Logs: /work/checkpoints/d07-{red,smoke,core,integration}-20261015.txt, d07-core-20261015.xml and d07-model-parity-20261015.json.

## Scope and remaining work

No new research fit or holdout scoring was run; Oracle remains OFF. Integration tests perform their existing bounded fits. This correction makes helper diagnostics agree with the existing model path; it does not establish a cause for holdout oversegmentation or improve frozen scores.

The unrestricted historical suite was not rerun. The five environment/identity-dependent evidence-test failures documented in distribution-fixes-20261015.md remain open; this checkpoint does not claim that the unrestricted suite is green. CUDA was not tested.

The next research step is a preregistered train/validation-only investigation of duration learning, emission fit and optimizer progress, preserving the existing test evidence.
