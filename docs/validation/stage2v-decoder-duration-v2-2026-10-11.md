# Stage 2V.8: corrected v2 decoder and episode-duration diagnostic

Date: 2026-10-11. Parent: `7dd6a26`. This is a short diagnostic, not full recovery acceptance.

## Protocol

Corrected 60,000-row v2 data, SHA256 `16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`; chronological first 60% training and last 20% evaluation. Training-only feature/context standardization. Seeds 901,902,903; 8 train and 4 evaluation windows of 240 rows, 4 joint iterations, 2 transition refinement steps, variance-aware kmeans emissions. One unchanged full context NHSMM fit per seed. No label-guided training or model selection. Three single-threaded workers. Hard runner timeout 300 seconds; observed elapsed 40.487 seconds; exit 0, empty stderr.

Reproduce with the shared Python environment and `PYTHONPATH=/workspace`:

```bash
python scripts/diagnose_trading_decoder_duration.py \
  --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz \
  --output /data/workspace/stage2/decoder-duration-v2-2026-10-11.json
```

## Same-model decoder comparison

| Median across seeds | Causal filter argmax | Retrospective Viterbi |
| --- | ---: | ---: |
| Accuracy | 79.27% | 81.77% |
| ARI | 0.5600 | 0.6135 |
| Boundary-F1 ±5 | 0.1842 | 0.2917 |
| State-run duration relative error | 92.25% | 84.11% |
| State changes | 68 | 40 |

The evaluation windows contain only 8 true state changes. Decoder agreement is 94.90–95.42%, yet the relatively few differing bars cause many extra boundaries. Viterbi reduces fragmentation but still fails the boundary (0.45) and duration (50%) thresholds. It uses the whole evaluation window and is not available as a live filtered signal. Existing causal acceptance remains unchanged.

The filter measurements reproduce the preceding corrected-v2 pilot, which validates that this comparison did not silently change its data, fit budget or primary arm.

## Learned episode durations

At zero standardized context and inference temperature 1, the learned episode medians are 31 bars for bull/bear and 32 for range/chaotic in every seed. D+ bucket mass is only 1.44–1.57%; tail end hazard is 0.468–0.500. A long-tail capability therefore exists but receives little mass in these short fits.

The evaluation state-run medians are bull 50, bear 40, range 139.5 and chaotic 25.5 bars, including window-edge censored runs. Separately counting only fully observed ground-truth episodes gives bear 40 (one episode), range 139.5 (two), chaotic 24 (one) and no complete bull episode. These tiny counts do not establish duration recovery or an accurate population duration estimate.

Learned episode distributions and decoded state runs are distinct: self-transitions can start new episodes without changing the visible state. Do not directly interpret the existing run-duration error as parameter recovery. The new diagnostic keeps both quantities explicit and excludes edge-censored episodes from its additional ground-truth episode summary.

## Finding and next hypothesis

Inference fragmentation explains part of the weakness, and the short-trained episode distributions remain near the uniform initialization's 32-bar scale. This is evidence for a duration-learning/initialization investigation, not proof of a recursion defect or of a globally optimal initializer. A next controlled experiment should compare unsupervised duration initialization or training schedules on identical corrected data and causal evaluation, without tuning against held-out labels.

No package code, package defaults or benchmark acceptance thresholds changed. Added diagnostic helpers are checked against explicit geometric masses and a censoring/self-transition example. Full package acceptance and the full corrected-v2 recovery protocol remain open.
