# Research ownership

NHSMM v3 owns ModelConfig, distributions, context encoding, initialization,
optimization, filtering/forecasts and core artifacts. Runtime dependencies:
NumPy and PyTorch.

Nautilus owns evaluation policy and research:
- adapter.qlib.nhsmm_validation: health/context evidence and snapshots;
- adapter.qlib.nhsmm_research: benchmark/Oracle and dataset helpers;
- tests/ml_stack/nhsmm: model/reference correctness tests;
- tests/ml_stack/fixtures/nhsmm: frozen mechanism-level test inputs.

The core never imports the host. Host diagnostics are coupled to the coordinated
core source version. Install pinned editable sources for integrated work.
Oracle controls remain opt-in. Research results do not confer live approval.
