# Build Log

## 2026-06-27 — Phase 1: Repository Setup & Dev Environment

### Tasks completed

- 1.1 Created the Git repository, README, Python `.gitignore`, and root commit.
- 1.2 Created the named package directory tree and `__init__.py` markers.
- 1.3 Created `requirements.txt`, a Python 3.11 virtual environment, and installed the core dependencies.
- 1.4 Added and installed Torch, Transformers, Hugging Face Hub, AutoGPTQ, and AutoAWQ for the CPU-only environment.
- 1.5 Created the four empty, valid JSON configuration skeletons.
- 1.6 Created the roadmap-prescribed placeholder import-isolation test.

### Files created or modified

- `README.md` — project identity and current Windows development setup.
- `.gitignore` — Python, virtual environment, build, secret, runtime, and editor exclusions.
- `requirements.txt` — Phase 1 runtime, backend, and test dependencies.
- `core/__init__.py` — core package marker.
- `engines/__init__.py` — execution backend package marker.
- `cluster/__init__.py` — cluster package marker.
- `cli/__init__.py` — CLI package marker.
- `ui/__init__.py` — desktop UI package marker.
- `notebooks/__init__.py` — notebook adapter package marker.
- `telemetry/__init__.py` — telemetry package marker.
- `config/__init__.py` — configuration package marker.
- `build/__init__.py` — build tooling package marker.
- `tests/__init__.py` — test package marker.
- `docs/__init__.py` — documentation package marker.
- `config/decision_matrix.json` — empty decision matrix skeleton.
- `config/persona_presets.json` — empty persona preset skeleton.
- `config/telemetry_thresholds.json` — empty telemetry threshold skeleton.
- `config/validation_suite.json` — empty validation suite skeleton.
- `tests/test_import_isolation.py` — Phase 1 `assert True` placeholder required by the roadmap.
- `docs/BUILD_LOG.md` — verified Phase 1 build record.
- `docs/ARCHITECTURE.md` — current structure, responsibilities, limitations, and run instructions.

### Verification commands and actual results

- `git log --oneline -1` returned `6309a0d Initial repository setup`.
- A PowerShell check of every named directory and its `__init__.py` returned `dir=True, init=True` for `core`, `engines`, `cluster`, `cli`, `ui`, `notebooks`, `telemetry`, `config`, `build`, `tests`, and `docs`.
- `.\.venv\Scripts\python.exe -c "import fastapi, redis, jwt; print('fastapi, redis, jwt imports: OK')"` printed `fastapi, redis, jwt imports: OK`.
- `.\.venv\Scripts\python.exe -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available())"` printed `torch version: 2.3.1+cpu` and `cuda available: False`.
- `.\.venv\Scripts\python.exe -m json.tool` exited successfully for all four configuration JSON files; each was reported as valid JSON.
- `.\.venv\Scripts\python.exe -m pytest tests/test_import_isolation.py -q` returned `1 passed in 0.02s`.
- `.\.venv\Scripts\python.exe -m pip check` returned `No broken requirements found.`

### Deviations from the roadmap

- Task 1.2 says “all 10 folders” but names 11 folders. All 11 named folders were created, because the explicit list is the less ambiguous requirement.
- AutoAWQ's normal isolated build could not see the already-installed Torch package. It was installed with `--no-build-isolation`; pip selected the Windows-compatible AutoAWQ 0.2.6 distribution and Torch 2.3.1+cpu. `pip check` confirmed a consistent environment.
- `pytest` was added to `requirements.txt` because Task 1.6 requires running pytest and the roadmap's first-command appendix includes it.
- The primary pasted-text attachment was unavailable on disk. The two readable, identical instruction files and `HaradiBots_Build_Tasks.docx` were used as the authoritative instructions.

### Needs manual verification

None for Phase 1. The user explicitly selected a CPU-compatible installation, and all Phase 1 checks were executable locally.

### Questions asked and answers

- Asked for authorization to use shell/Git and initialize `G:\HB QUANTRA`; the user authorized it.
- Asked whether to install CPU-compatible ML dependencies after `nvidia-smi` was unavailable; the user replied, “Install CPU-compatible ML dependencies.”

## 2026-06-27 — Phase 2: Core Schemas & Authentication

### Tasks completed

- 2.1 Defined strict Pydantic v2 models for the v3.0 inbound, outbound, hardware, strategy, error, teardown, model metadata, and validation contracts.
- 2.2 Implemented the SHA-256 API-key credential store loader and validator.
- 2.3 Implemented HS256 JWT validation with required signature, expiry, audience, and subject checks.
- 2.4 Implemented the schema-versioned authentication gate, request-scoped verified identity, and structured 401 errors.
- 2.5 Added the development credential generator and verified the user-generated key against its stored hash.

### Files created or modified

- `core/schemas.py` — strict, extra-forbidden Pydantic v2 contract models.
- `core/auth_middleware.py` — API-key and JWT validators plus the main authentication gate.
- `scripts/gen_credentials.py` — local API-key generator that never prints the plaintext.
- `README.md` — local authentication setup and secret-handling instructions.
- `docs/BUILD_LOG.md` — verified Phase 2 build record.
- `docs/ARCHITECTURE.md` — current structure, contracts, authentication flow, variables, and run instructions.
- `.env` — ignored local plaintext development key, created by the user.
- `config/credentials.json` — ignored local SHA-256 credential store, created by the user.

### Verification commands and actual results

- `.\.venv\Scripts\python.exe -c "from core.schemas import JobEnvelope"` imported successfully and printed `JobEnvelope`.
- An inline API-key check authenticated the generated valid key as `local-development`; `definitely-invalid` raised `AuthError`.
- An inline JWT check returned `verified-user` for a valid signed token; an expired token raised `AuthError`.
- An inline `authenticate()` check on an envelope with `auth=None` returned `ErrorEnvelope 401 authentication_failed`.
- A redacted generator audit confirmed `.env` contains `DEV_API_KEY`, the corresponding SHA-256 is present in `credentials.json`, no plaintext key field exists in the store, and authentication passes.
- `.\.venv\Scripts\python.exe -m pytest tests/test_import_isolation.py -q` returned `1 passed in 0.03s`.
- `.\.venv\Scripts\python.exe -m pip check` returned `No broken requirements found.`
- `git diff --check` exited successfully.

### Deviations from the roadmap

- The build-task DOCX references Architecture Spec §1.2 without reproducing its fields. Work paused until the user supplied `HaradiBots_Architecture_v3.0.docx`; its §1.2 table then became the authority for the exact top-level wire fields.
- §1.2 specifies the exact `JobEnvelope` and `ProgressEvent` fields but describes several nested payloads semantically. Their typed fields were derived from those descriptions and the later hardware, strategy, teardown, model-inspection, and validation sections of the same v3.0 architecture document.
- `JobEnvelope.auth` is represented as nullable at model-construction time so the required Task 2.4 missing-auth request can reach `authenticate()` and receive a structured 401. The authentication gate still rejects it before orchestration.

### Needs manual verification

None for Phase 2. The user ran the required secret generator, and all checks were executable locally without exposing secret values.

### Questions asked and answers

- Asked for Architecture Spec §1.2 because the build-task document omitted the binding field table; the user supplied `HaradiBots_Architecture_v3.0.docx`.
- Asked the user to run `scripts/gen_credentials.py` so the assistant would not generate or expose the plaintext secret; the user ran it and supplied the non-secret success message.

## 2026-06-27 — Phase 3: Hardware Profiler

### Tasks completed

- 3.1 Implemented NVIDIA GPU enumeration and telemetry through NVML with graceful CPU fallback.
- 3.2 Implemented RAM, physical-core, topology, clock, and conservative ISA profiling.
- 3.3 Implemented P-core-only hybrid pinning, uniform-core minus-one selection, and degraded fallback.
- 3.4 Implemented the exact weight, GQA-aware KV-cache, and partial-offload formulae from Architecture §2.2.
- 3.5 Populated all eight Architecture §2.3 matrix rows and implemented deterministic strategy selection with model-size limits.
- 3.6 Implemented UUID- and UTC-identified snapshots that validate through `HardwareProfile`.

### Files created or modified

- `core/profiler.py` — GPU/CPU profiling, thread planning, memory formulae, matrix loading, strategy selection, and top-level snapshot.
- `config/decision_matrix.json` — all eight architecture-defined hardware tiers and machine-readable model limits.
- `requirements.txt` — explicit `nvidia-ml-py3` and `psutil` profiler dependencies.
- `README.md` — hardware snapshot usage and fallback behavior.
- `docs/BUILD_LOG.md` — verified Phase 3 build record.
- `docs/ARCHITECTURE.md` — current structure, profiler responsibilities, contracts, limitations, and run instructions.

### Verification commands and actual results

- `snapshot_gpu()` returned an empty list on this no-NVIDIA environment without raising.
- `snapshot_cpu()` returned a dictionary with 11.874 GiB total RAM, 2 physical cores, `core_topology='uniform'`, and a 3.5 GHz P-core/fallback clock during the final acceptance run.
- Synthetic hybrid, uniform, and unknown topology inputs verified P-core IDs `[0, 2, 4, 6]`, uniform thread count `7`, and degraded fallback thread count `7` with its warning flag.
- Formula checks returned 14,000,000,000 bytes for FP16 7B, 3,500,000,000 bytes for Q4 7B, 536,870,912 bytes for the test GQA cache, and 16 partial-offload layers for the hand-calculated test.
- The matrix audit found eight rows; a 4 GiB GPU with a 7B/32-layer model selected `Q4_K_M`; a 13B model on that tier raised `ValueError`.
- `snapshot()` validated through `HardwareProfile`, returned `gpu_count=0`, and reported a uniform CPU topology.
- `.\.venv\Scripts\python.exe -m json.tool config/decision_matrix.json` exited successfully.
- `.\.venv\Scripts\python.exe -m pytest tests/test_import_isolation.py -q` returned `1 passed in 0.04s`.
- `.\.venv\Scripts\python.exe -m pip check` returned `No broken requirements found.`
- `git diff --check` exited successfully.

### Deviations from the roadmap

- Architecture §2.3 contains eight named rows although the phase wording groups them loosely. All eight explicit rows were preserved.
- The matrix's textual ranges were supplemented with numeric minimum and maximum model-size fields so selection can reject an impossible oversized model deterministically.
- Windows hybrid classification uses the native `GetSystemCpuSetInformation` efficiency-class topology API. This exposes the OS-supported P/E distinction without adding a third-party CPUID dependency.
- ISA reporting is conservative on Windows: unsupported flags are omitted rather than inferred from a CPU product name.

### Needs manual verification

- On an NVIDIA system with a working NVML driver, run `python -c "from core.profiler import snapshot_gpu; print(snapshot_gpu())"` and confirm every device reports UUID, VRAM, compute capability, bandwidth, temperature, power draw/limit, and NVLink peers as applicable.
- On a real hybrid P/E-core Windows or Linux CPU, run `python -c "from core.profiler import snapshot_cpu, get_thread_config; p=snapshot_cpu(); print(p); print(get_thread_config(p))"` and confirm `p_core_ids` and `e_core_ids` are nonempty and only P-core IDs are selected.

### Questions asked and answers

None.
