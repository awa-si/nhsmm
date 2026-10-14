# Joint Gaussian/duration recovery — completed frozen study

Jointly learning scalar Gaussian means, variances and fixed-bin durations
recovers all four duration laws on all three preregistered seeds. Freezing
the same observation-initialized emissions fails on all three seeds, despite
correct likelihoods and gradients. This separates emission misspecification
from a duration-recursion defect in this controlled experiment.

## Contract and evidence

Frozen source: `ac312d6a297d25a1330baf7ff411971c7e16429f`.
The [protocol](joint-gaussian-duration-protocol.json) was committed before fits;
its `predeclared_not_executed` status is retained as an immutable declaration.
Completion is recorded by the raw reports, independent proofs and
[SHA-bound batch](results/joint-gaussian-duration-batch-native-20261014.json).

Seeds1930/1931/1932 each use1024 training and24 outer sequences of300 rows,
Gaussian sigma2, four states and positive support on all ages1..260. Duration
bins have width8 (last bin width4). Static transition/initial laws are known.
All arms start with uniform durations; estimated frozen/joint arms share the
same training-only Lloyd initialization. Each arm performs30 actual EM updates:
270 total. No seed, family, initialization, budget or gate changes follow outer
outcomes. Oracle OFF; no latent labels or true-law support enter these fits.
Known-emission control explicitly uses known generating emission parameters
and is a separate information reference, not an initialization-matched arm.

## Results

Maxima below cover all four states and all three seeds.

| Arm | Full gates | Maximum duration TV | Maximum mean error | Maximum variance relative error |
| --- | ---: | ---: | ---: | ---: |
| Known-emission control | 3/3 | 0.100810 | 0 | 0 |
| Estimated emissions, frozen | 0/3 | 0.958354 | 0.922510 | 0.658552 |
| Estimated emissions, jointly learned | 3/3 | 0.100770 | 0.017221 | 0.016725 |

Gates are duration TV<=0.12, each mean error<=0.5, each variance relative
error<=0.25, positive outer gain and passing numerical checks.

| Seed | Joint minus frozen outer LL, nats | Joint outer gain, nats/row |
| --- | ---: | ---: |
| 1930 | +2389.485932 | +0.439628733 |
| 1931 | +2338.011481 | +0.438753430 |
| 1932 | +2562.219381 | +0.461024426 |

Joint versus frozen is the primary matched comparison: identical data,
initialization, duration family and30-update budget. Each reported outer gain
uses uniform duration with that arm's initial emissions; joint gain includes
emission learning and is not a duration-only gain. Positive gain alone is
insufficient: even the failed frozen arm improves over its poor baseline.

## What this explains

The hard observation-cluster initialization produces shifted means and
variances roughly31–66% below the generating variance. Keeping those emissions
fixed leads to learned mean durations around8–37 rows and duration TVs0.73–0.96.
Allowing soft-posterior emission updates restores means, variances and durations
to the preregistered gates on3/3 seeds, close to the known-emission reference.

The supported mechanism is that overly narrow, shifted emission densities
favor excessive state switching, and duration fitting compensates with short
episodes. This interpretation is consistent with the matched-arm results;
this study does not separately isolate mean bias from variance bias. It shows
that a correct HSMM recursion can still learn incorrect durations under bad
fixed emissions. It does not establish the cause of every production NHSMM
or neural-context failure.

## Verification and remaining boundary

All nine fits pass independent terminal training LL on the full training set
and independent outer LL on all24 outer sequences. Public autograd checks all
duration-bin, mean and raw-variance gradients; public filtering and independent
completed-segment recursion are compared on two complete outer sequences per
arm at temperature1. LL/gradient errors are below1e-7; filtering below1e-8.
Every augmented EM history is monotone within1e-6. Final resume is byte-preserving
and performs zero updates. All three raw reports and proofs are SHA-bound.

The52-file execution manifests are supplemented by a [complete static closure
audit](results/joint-source-closure-audit-native-20261014.json):54 files match
frozen Git blob identities and SHA256s, including two additional import owners.
The23 focused tests, Ruff, Black, compilation and both bounded runner smokes pass.
Local scaled-backward arithmetic is about2.07x faster than the independent old
log-duration E-step on16x300; this is not a neural/production speed claim.

The next separate boundary is learning transition and initial parameters before
adding neural context. Production optimizer defaults were not changed here.
There is no Trading, neural-context, legacy NPZ or global-convergence acceptance.
