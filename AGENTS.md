# Agent Guidance

Repository-wide AI instructions are defined in [`agent.md`](./agent.md).

Before analyzing, designing, reviewing, documenting, or modifying NHSMM:

1. read `agent.md` from the current `develop` branch;
2. treat it as the authoritative repository-specific operating contract;
3. inspect the current implementation before relying on historical docs, scripts, or tests;
4. preserve probabilistic correctness, explicit-duration HSMM semantics, tensor-shape contracts, masking, numerical stability, and temporal/causal validity;
5. use GitHub Patch as the preferred repository edit workflow when the plugin/skill is available.

This repository is a Python/PyTorch probabilistic sequence-modeling library. It is not a React/TypeScript/Vite project.
