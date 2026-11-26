# Linux x86-64 NHSMM native Rust libraries

Precompiled, stripped cdylibs for direct source checkout access, from the
workspace's previously compiled HMM and HSMM kernels.

Platform: Linux x86-64, GNU glibc >=2.34; native ABI is not portable.
Other hosts use packaged Rust source build or the Python backend. See
`nhsmm/native/prebuilt/linux_x86_64_glibc/manifest.json` for source/binary
SHA-256 digests and provenance notes. Runtime loads the verified package copy
via platform and checksum gates; never infer cross-platform compatibility.

Not a reproducible-build attestation, security audit, or live-trading release.
