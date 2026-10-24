# Explicit stationary state/age entry

Default filtering still starts a fresh episode at age one.
For an unknown-age observation window under a static time-homogeneous model:

    from nhsmm import HSMMFilterRuntime, stationary_filter_prior
    from nhsmm.filtering import filter_model_sequence

    prior = stationary_filter_prior(model)
    batch = filter_model_sequence(model, observations, initial_prior=prior)
    runtime = HSMMFilterRuntime(model, initial_prior=prior)

This is an unconditional latent entry prior, before the first emission. It
replaces the learned fresh-episode initial pi for this inference call. It does
not infer a true age from labels or assert that financial regimes are stationary.
It is unavailable for neural/context-varying models. An explicit normalized
HSMMFilterState may also supply a different caller-owned joint entry prior.

Let q_i(a) be the static total duration PMF (last mass is D+ survival when enabled)
and A(i,a,j) the exit destination kernel. The episode-entry kernel is
B(i,j)=sum_a q_i(a)*A(i,a,j). Solve nu*B=nu. Unnormalized age occupancy is
nu_i*S_i(a); for the D+ bucket it is nu_i*S_i(D)/h_i. Normalize over state and age.
This accounts for length-biased sampling of ongoing episodes and age-dependent
destinations. Only a K-by-K solve is needed, not a dense KD-by-KD solve.
Nonunique or numerically invalid stationary laws are rejected.

The input prior must match model dtype, device, K and D; batch one broadcasts.
Runtime reset preserves this chosen prior. Recompute it and construct a fresh
runtime after changing model weights or temperature; pass the same temperature
to the prior builder and filter. No caller-owned buffers are retained by alias.

Fresh-boundary training and stationary-entry inference are distinct contracts.
The ordinary likelihood and supervised score APIs retain their fresh-episode
semantics. Core artifact v3 stores the learned model; the application manifest
must record its inference entry policy and reconstruct the prior after loading.
