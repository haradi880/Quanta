# Release Status

Product: **HaradiBots Quanta**. Windows executable: **`Quanta.exe`**.

This matrix is the authoritative production-acceptance record. “Passed” means
the requirement was exercised against the real implementation; unit tests or
static inspection alone are labeled separately.

| Gate | Status | Current evidence |
|---|---|---|
| v3.1 contracts and authenticated lifecycle | Passed | Strict schema, FSM, auth, and lifecycle tests |
| CPU GGUF inference | Passed | Bundled `llama-completion` generated a real tiny-model continuation through `GGUFWorker` |
| CUDA GGUF inference | External verification required | Current Windows host has no NVIDIA driver or `nvidia-smi` |
| Full-precision conversion/quantization | Passed | Bundled quantizer converted F32 GGUF (93.11 MiB) to Q4_0 (17.50 MiB) |
| Three-domain perplexity | Passed | Real logic, retrieval, and code reference/candidate evaluation |
| Golden-prompt reporting | Passed | Real golden item emitted alongside composite scoring |
| Poor/critical delivery control | Passed | Real degraded Q4 candidate was classified critical and quarantined |
| Persistence and restart recovery | Passed | Every authenticated lifecycle is persisted; startup marks abandoned rows interrupted; restart behavior is automated-tested |
| Cancellation and forced-kill teardown | Passed by automated integration test | Worker tree harvesting and forced-kill coverage |
| Five-phase purge | Passed by automated integration test | Ordering, unsafe-root, symlink, CLI, and GUI confirmation coverage |
| Local Garnet lifecycle | Passed | Real bundled `PING`, `HSET`, `HGETALL`, `SCAN`, and clean stop |
| Vendor integrity and notices | Passed | Six required assets plus full SHA-256 inventory; Garnet/.NET notices bundled |
| One-dir executable build | Passed on build host | Packaged executable, packaged doctor, and local-GGUF CLI lifecycle passed |
| Clean-machine offline execution | External verification required | Build-host local-GGUF lifecycle passes; run `build/verify_offline_release.ps1` on a second disconnected Windows machine |
| Security/dependency checks | Passed and CI-enforced | pip-audit, Bandit medium/high scan, and tracked-production secret signatures |
| Core branch coverage | Passed | 80% minimum enforced locally and in CI |
| Optional Ray/SLURM/Kubernetes | Foundation only | Provisioning/readiness adapters tested; distributed execution and artifact return are not integrated |

No release tag should be created while any standalone-required row says
“External verification required.” Optional cluster rows do not block the
single-machine v1 release, but must not be marketed as production-complete.
