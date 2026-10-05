# Nautilus state-count usability: 3 vs 4 states — 2026-10-05

## Scope

Downstream empirical usability check on real Nautilus observations. This is not package-core proof and does not introduce trading assumptions into NHSMM itself.

- repository head under test: `c41bc19737d7010f00bcecd936875e9ac6a840dd`
- source dataset: repository-declared AWA dataset `nautilus`, mounted read-only at `/data/nautilus`
- observation contract: 18-dimensional Nautilus -> NHSMM interface contract
- fit API: public `NHSMM.optimize()`, `predict()`, `log_likelihood()`
- causal: yes
- `max_duration=12`
- `max_iter=8`
- `n_init=1`
- `emission_init_mode="kmeans"`
- `dropout=0.0`
- comparison: 3 states vs 4 states
- replication: 3 seed bases x 2 chronological OOS folds = 6 runs per state count

The dataset is not copied into the repository or workspace. `workspace.ini` declares `data = nautilus`; AWA mounts it read-only for executions.

## Results

| metric | 3 states | 4 states |
| --- | ---: | ---: |
| runs | 6 | 6 |
| healthy fraction | **1.0000** | 0.8333 |
| mean OOS log-likelihood / step | **-13.074227** | -13.080162 |
| mean train/OOS generalization gap | 0.051264 | **0.049978** |
| mean effective states | 2.7878 | **3.8317** |
| minimum Viterbi states used | **2** | 1 |
| mean Viterbi states used | 2.0 | 2.0 |
| minimum posterior-MAP states used | **2** | 1 |
| mean posterior-MAP states used | 2.0 | **2.6667** |
| mean Viterbi switch rate | **0.00599** | 0.00908 |
| mean Viterbi run length | 155.14 | 236.33 |
| mean posterior entropy | 0.9828 | 1.3229 |
| mean posterior margin | **0.2184** | 0.1055 |

4 minus 3:

- OOS LL: `-0.005935` per timestep
- effective states: `+1.04387`
- healthy fraction: `-0.16667`
- mean Viterbi states: no gain

## Run-level observation

3-state:
- 6/6 runs healthy
- every Viterbi decode used exactly 2 states
- every posterior-MAP decode used exactly 2 states
- the third state remained soft/non-dominant but contributed to posterior occupancy, with mean effective-state count 2.79

4-state:
- 5/6 runs healthy
- one run collapsed to a single dominant state in both Viterbi and posterior-MAP
- other runs used 2-3 dominant posterior states
- posterior effective-state count was high (~3.83), but that extra soft-state capacity did not improve OOS likelihood or hard-state robustness

Representative failed 4-state run:

```text
K=4 seed=1001 fold-1
OOS=-13.091073
healthy=False
effective_states=3.777
Viterbi=[0.0, 0.0, 1.0, 0.0]
posterior_MAP=[0.0, 0.0, 1.0, 0.0]
```

## Conclusion

For the current Nautilus 18-D observation contract, `n_states=3` remains the preferred operational baseline.

The 4-state model uses more posterior capacity, but this does not translate into better OOS likelihood or more robust dominant-regime decoding. It is therefore not justified as the default by the current evidence.

The third state in the 3-state model should not be forcibly made Viterbi-dominant. Prior diagnostics showed that:
- it is not simply the closest/redundant merge candidate
- transition-entropy, posterior-entropy, emission-separation, and accessibility regularizers did not restore robust three-state hard decoding
- slowing emission+duration jointly can restore the third Viterbi state only at a material OOS likelihood cost
- Viterbi and posterior marginal MAP both typically select two dominant states

Interpretation: the 3-state model currently provides useful soft latent capacity while expressing roughly two dominant OOS regimes.

## Re-run

Re-run after material changes to:
- emission/duration parameterization
- joint training objective
- restart selection
- causal filtering/Viterbi semantics
- Nautilus 18-D observation mapping
- state-count defaults
