# Independent HMM, HSMM and NHSMM modes

Code anchor: doc:nhsmm.model-modes.

Choose one mode at construction; the same public training, artifact and runtime
APIs work for all three. A mode change builds and fits a new model. It does not
reinterpret an existing fitted artifact or migrate optimizer state.

```python
from nhsmm import ModelConfig, NHSMM, HSMMFilterRuntime

mode = "hmm"  # "hmm" -> "hsmm" -> "nhsmm"
config = ModelConfig.for_mode(mode, n_states=3, n_features=3, seed=7)
model = NHSMM(config, device="cpu")
model.initialize_distributions()
# model.fit_supervised(observations, state_labels, steps=200)
model.eval()
runtime = HSMMFilterRuntime(model)
```

| Mode | Duration and transitions | Neural components |
|---|---|---|
| hmm | One-step latent updates; ordinary self transitions; no duration tail | No encoder or context networks |
| hsmm | Explicit durations with max_duration >=2; no self-reset transitions | No encoder or context networks |
| nhsmm | Same explicit-duration law, with neural context modulation | Internal causal encoder by default, or explicit external context |

The HMM's state-run dwell times are geometric through self transitions; the
internal one-step duration support does not force the observed state to change
each row. HSMM state labels identify episodes because self resets are excluded.
Finite durations and the existing geometric D+ tail are available in HSMM/NHSMM.
Set duration_tail=True only for those modes.

Static profiles reject context/hidden dimensions, encoder enablement and
incompatible duration settings. All profiles require causal=True.
HSMM/NHSMM profiles require the existing ergodic no-self-transition K>=2
contract. Explicit conflicting overrides fail; no requested option is silently
discarded. Pass max_duration only for HSMM/NHSMM, since HMM requires exactly one.

For externally supplied context, use:
```python
config = ModelConfig.for_mode(
    "nhsmm", n_states=3, n_features=3, context_dim=24, max_duration=8
)
```
This builds context heads without an observation encoder. To request an internal
encoder with a declared output width, additionally set use_context_encoder=True.
Emission context must precede the observation; outgoing boundary context applies
to the next transition. Use BASE_CONTEXT or the documented CausalContext
availability mask for missing external context, preserving a consistent runtime
context mode. Never feed future/unfinished timeframe bars.

model.config.model_mode reports the resolved architecture, not a separately
stored flag that can disagree with the parameters. A plain ModelConfig and its
with_overrides remain the low-level customization API; deliberately changing
architecture or injecting an encoder there is outside the selected profile.
Old low-level construction defaults remain unchanged.

Artifacts remain schema version3: the complete resolved existing fields encode
the model architecture, so no new serialized field or inference kernel is needed.
Loading reconstructs the same context-free or neural structure. Freeze/reset,
causality, provenance and source-version requirements remain in production.md.

Canonical tests: tests/ml_stack/nhsmm/test_model_modes.py in Nautilus.
They check absence of neural networks in static modes, independent raw-weight
path enumeration, supervised improvement without state collapse on a bounded
separable fixture, artifact/stream/retraining behavior, availability fallback,
and strict option validation. Synthetic usability is not market/trading value.

Reference: Johnson and Willsky (2013), section3, explicit-duration state modeling
and why excluding self transitions makes durations describe state episodes:
https://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf
We reuse the package's independently verified duration/filtering/optimizer kernels;
this change adds a configuration selection boundary, not a new HSMM algorithm.
