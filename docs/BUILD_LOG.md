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
- The final acceptance rerun printed `Phase 5 final acceptance audit: PASS`.
- A tracked import scan found no `core/` or `engines/` imports from `ui`, `cli`, or `notebooks`.

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

## 2026-06-27 — Phase 5: Orchestrator & State Machine

### Tasks completed

- 5.1 Implemented the eight-state FSM and explicit event transition table.
- 5.2 Implemented registered-root and recursive child-process harvesting with TERM, three-second grace, KILL escalation, manifest emission, and forced-kill warnings.
- 5.3 Implemented the authenticated async job generator, state progress, structured failure events, and unconditional teardown after state-machine entry.
- 5.4 Implemented per-job worker registration, immutable handle inspection, and registry cleanup.
- 5.5 Implemented concrete auto strategy planning and full-formula manual OOM warnings.

### Files created or modified

- `core/orchestrator.py` — FSM, worker ownership, process-tree teardown, job routing, and module-level interface entry point.
- `core/accelerator.py` — auto/manual strategy planner and conservative manual VRAM validation.
- `core/schemas.py` — approved `ModelMetaProfile.parameter_count` contract correction.
- `core/hf_inspector.py` — exact SafeTensor parameter-total extraction from Hub metadata.
- `README.md` — common orchestrator entry-point usage and Phase 6 worker boundary.
- `docs/BUILD_LOG.md` — verified Phase 5 build record.
- `docs/ARCHITECTURE.md` — current structure, orchestration flow, planner, contracts, limitations, and run instructions.

### Verification commands and actual results

- FSM verification accepted `IDLE + job_received → PROFILING` and raised `StateMachineError` for `IDLE + execution_complete`.
- A synthetic worker that ignored TERM was force-killed after the injected test grace; `teardown_complete` reported `forced_kill_count=1`, and a warning was logged.
- A valid-auth job with a missing local model path produced `IDLE → PROFILING → PLANNING → ERROR → TEARDOWN → IDLE`; teardown was the event immediately before terminal completion.
- A registered test worker plus live Llama-3 metadata produced `IDLE → PROFILING → PLANNING → EXECUTING → VALIDATING → TEARDOWN → IDLE`; its registry entry was empty afterward.
- Auto planner checks returned the architecture-defined backend for all eight hardware tiers.
- A 4 GiB GPU manual override to FP16/all 32 layers returned `warning=True`, predicting 19,127,461,755 bytes against 4,294,967,296 available bytes.
- Live `NousResearch/Meta-Llama-3-8B` inspection returned exact `parameter_count=8030261248` and validated through `ModelMetaProfile`.
- Recursive teardown regression returned one forced kill and an empty registry.
- `.\.venv\Scripts\python.exe -m compileall -q core` exited successfully.
- `.\.venv\Scripts\python.exe -m pytest tests/test_import_isolation.py -q` returned `1 passed in 0.02s`.
- `.\.venv\Scripts\python.exe -m pip check` returned `No broken requirements found.`
- `git diff --check` exited successfully.

### Deviations from the roadmap

- Phase 5 needs exact model size for the Decision Matrix, but the Phase 2 metadata contract omitted it. With explicit user approval, `parameter_count` was added to `ModelMetaProfile` and populated from the Hub's exact SafeTensor total instead of guessing from names or file sizes.
- The appendix lists `core/accelerator.py` before `core/orchestrator.py`, while the numbered Phase 5 tasks place accelerator last. The numbered task order was followed, and the orchestrator was switched to the accelerator planner immediately in Task 5.5.
- Worker execution is a real registered-handle protocol, but concrete backend classes intentionally remain absent until Phase 6. A valid job without a worker fails structurally at EXECUTING rather than reporting false success.

### Needs manual verification

- Real TERM→KILL escalation and descendant discovery must be verified with actual backend subprocess trees in Phase 6.
- Ray actor harvesting must be integration-tested when Ray handles are introduced in Phase 11.

### Questions asked and answers

- Asked permission to add `parameter_count` to `core/schemas.py` and populate it in `core/hf_inspector.py`, because these files were outside Task 5.3's listed key file. The user replied `approved`.

## 2026-06-27 — Cumulative Audit: Phases 1–5

- Rechecked every named Phase 1 directory and configuration JSON.
- Rechecked generated API-key authentication, invalid-key rejection, JWT validation, and structured missing-auth failure.
- Rechecked CPU/GPU fallback profiling, schema validation, thread selection, and all three memory formulae.
- Rechecked live Llama-3 metadata, exact parameter count, and GQA ratio.
- Rechecked FSM transition rejection, strategy planning, manual OOM warning, missing-model error path, mandatory teardown, and registry cleanup.
- `pytest tests -q` returned `1 passed in 0.04s`; `pip check` returned no broken requirements; Git was clean.
- The cumulative script printed `CUMULATIVE PHASE 1-5 AUDIT: PASS`.

## 2026-06-27 — Phase 6: Execution Backends

### Tasks completed

- 6.1 Implemented the abstract lazy-loading worker contract.
- 6.2 Implemented GGUF execution solely through a configured llama.cpp subprocess with line-by-line progress parsing.
- 6.3 Implemented guarded AutoAWQ quantization, inference validation, cleanup, and orchestrator registration.
- 6.4 Implemented guarded EXL2 conversion through the official conversion script plus ExLlamaV2 inference validation.
- 6.5 Implemented vLLM tensor-parallel loading, inference validation, and exact native `/tokenize` counting.
- 6.6 Implemented the Architecture §3.2 over-compilation safety table.

### Files created or modified

- `engines/base_worker.py` — backend-neutral async worker and lifecycle contract.
- `engines/gguf_worker.py` — llama.cpp subprocess execution and output parsing.
- `engines/awq_worker.py` — lazy AutoAWQ quantization and validation.
- `engines/exl2_worker.py` — lazy ExLlamaV2 conversion and validation.
- `engines/vllm_worker.py` — lazy tensor-parallel vLLM loading and native tokenizer client.
- `core/accelerator.py` — over-compilation safety guard.
- `core/orchestrator.py` — approved lazy backend factory, strategy path injection, immediate registry insertion, and worker-error propagation.
- `requirements.txt` — compatible Torch, Transformers, NumPy, and PEFT pins for AutoAWQ/AutoGPTQ.
- `README.md` — backend setup, lazy-loading behavior, and required binary/script variables.
- `docs/BUILD_LOG.md` — cumulative and Phase 6 verification record.
- `docs/ARCHITECTURE.md` — current backend structure, responsibilities, variables, and limitations.

### Verification commands and actual results

- Importing `BaseWorker` triggered none of Torch, Transformers, AWQ, EXL2, vLLM, or llama.cpp.
- A controlled real Python subprocess emitted `42% complete`; `GGUFWorker` produced launch, running, and completion events and parsed `progress_pct=42.0`.
- A missing GGUF binary/model produced one structured error and imported no `llama_cpp` binding.
- Simulated missing AutoAWQ and ExLlamaV2 imports each produced one structured backend error without a traceback.
- The orchestrator factory created an AWQ worker and inserted it into the job registry immediately.
- A fake vLLM class received `tensor_parallel_size=3`.
- A local HTTP tokenizer server received `{"prompt":"exact","model":"test/model"}` and `VLLMWorker` returned exact count `7`.
- All Q4-or-lower, AWQ, GPTQ, and EXL2 blocked conversion classes returned `allowed=False` with reasons; FP16→Q4, Q8→Q6, and Q6→Q4 returned `allowed=True`.
- Importing every worker module in a fresh process loaded no backend library.
- AutoAWQ and AutoGPTQ imports succeeded after dependency correction; `pip check` returned `No broken requirements found.`
- In-process worker teardown completed gracefully with zero forced kills.
- `.\.venv\Scripts\python.exe -m compileall -q core engines` exited successfully.
- `.\.venv\Scripts\python.exe -m pytest tests -q` returned `1 passed in 0.02s`.
- `git diff --check` exited successfully.
- The final AST audit found no backend import at worker-module scope; no Python `llama_cpp` import exists.
- The final Phase 6 script printed `PHASE 6 FINAL AUDIT: PASS`; controlled GGUF parsing returned `55%`.
- `pip install -r requirements.txt --dry-run` resolved every requirement without a planned change.

### Deviations from the roadmap

- With explicit user approval, `core/orchestrator.py` was updated even though it is not listed under the Phase 6 key files. This was required to instantiate the selected backend and satisfy immediate worker-registry insertion.
- The official EXL2 converter is a script, not a stable in-process conversion API. The worker lazily imports ExLlamaV2 for inference but launches configured `convert.py` with its documented `-i`, `-o`, `-cf`, and `-b` arguments.
- The unconstrained Phase 1 dependency set resolved incompatible releases. It was corrected to Torch 2.3.1, Transformers 4.45.2, NumPy 1.26.4, and PEFT 0.7.1; both AutoAWQ and AutoGPTQ then imported.
- EXL2 and vLLM were not installed because this CPU Windows environment lacks the platform-specific CUDA build requirements. Their missing-package paths are verified and structured.

### Needs manual verification

- Run GGUF execution with a real compiled llama.cpp binary and local Q4_K_M model.
- On a compatible NVIDIA system, install the matching AutoAWQ kernels and run a real quantization.
- Install the matching ExLlamaV2/CUDA build, configure its official `convert.py`, and run conversion plus inference.
- Install vLLM on a supported GPU host, launch a tensor-parallel model, and compare `count_tokens()` to the live server response.

### Questions asked and answers

- Asked permission to update `core/orchestrator.py` for automatic backend construction and registry insertion. The user replied `approved`.

## 2026-06-27 — Phase 4: HuggingFace Inspector

### Tasks completed

- 4.1 Implemented asynchronous repository existence, gated-state, access, and storage-size inspection.
- 4.2 Implemented the `.safetensors`, `.bin`, `.gguf`, and `.json` manifest plus shard-count detection.
- 4.3 Implemented config normalization, MHA/GQA/MQA classification, and KV-head ratio calculation.
- 4.4 Implemented tokenizer-template classification, model-family mapping, and config/filename quantization detection.
- 4.5 Made strict `ModelMetaProfile` validation the return boundary and added warnings for missing model fields.

### Files created or modified

- `core/hf_inspector.py` — metadata-only asynchronous Hugging Face inspector.
- `README.md` — inspector usage and gated-token guidance.
- `docs/BUILD_LOG.md` — verified Phase 4 build record.
- `docs/ARCHITECTURE.md` — current structure, inspector responsibility, contracts, variables, limitations, and run instructions.

### Verification commands and actual results

- Live inspection of `meta-llama/Llama-2-7b-hf` returned `repo_exists=True`, `is_gated=True`, and `repo_size_bytes=53909360564`.
- Its filtered manifest contained 11 positive-size files and detected two SafeTensor shards.
- Live inspection of `NousResearch/Meta-Llama-3-8B` returned 32 attention heads, 8 KV heads, `attention_type='gqa'`, and `kv_head_ratio=0.25`.
- Live inspection of `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` returned `is_prequantized=True`, `quant_format='awq'`, `model_family='mistral'`, and `chat_template_type='mistral'`.
- Both accessible Llama and Mistral outputs validated through `ModelMetaProfile`.
- Live inspection of `openai-community/gpt2`, whose config omits `num_key_value_heads`, returned `attention_type='mha'` with `upper_bound_only=True` and validated through `ModelMetaProfile`.
- The gated Meta Llama 2 metadata remained usable while inaccessible `config.json` and `tokenizer_config.json` emitted clear warnings and missing fields emitted field-specific warnings.
- `.\.venv\Scripts\python.exe -m compileall -q core` exited successfully.
- `.\.venv\Scripts\python.exe -m pytest tests/test_import_isolation.py -q` returned `1 passed in 0.02s`.
- `.\.venv\Scripts\python.exe -m pip check` returned `No broken requirements found.`
- `git diff --check` exited successfully.

### Deviations from the roadmap

- Meta's gated Llama 3 config requires an account with accepted access. The exact GQA check used the accessible `NousResearch/Meta-Llama-3-8B` mirror rather than requesting or exposing a user token.
- `tokenizer_config.json` in that mirror does not declare a chat template, so `chat_template_type` correctly remains null; no template was inferred from model family alone.
- Missing gated configuration is logged and represented conservatively instead of turning an otherwise valid repository-existence result into a crash.

### Needs manual verification

- With an account authorized for Meta Llama repositories, set `HF_TOKEN` privately and run `inspect_repo('meta-llama/Meta-Llama-3-8B')` to verify full config and tokenizer metadata on the original gated repository.

### Questions asked and answers

None.
## 2026-06-27 — Phase 7: Validation Engine

### Tasks completed

- 7.1 Added five logic prompts, five code prompts, a retrieval template, and weights summing to 1.0.
- 7.2 Implemented causal perplexity for likelihood and Torch model interfaces.
- 7.3 Implemented randomized linked-fact retrieval prompts near the evaluated context length.
- 7.4 Implemented weighted original-versus-quantized validation with separately scored golden prompts.
- 7.5 Implemented severity, confirmation, and quarantine policy mapping.

### Files created or modified

- `config/validation_suite.json` — validation prompts, templates, and weights.
- `core/accelerator.py` — perplexity, retrieval generation, suite runner, and severity mapping.
- `tests/test_validation.py` — formula, identical-model, poor, and critical regression tests.
- `requirements.txt` — Hugging Face `evaluate` reference package.
- `README.md`, `docs/ARCHITECTURE.md`, and `docs/BUILD_LOG.md` — Phase 7 usage and design records.

### Verification commands and actual results

- Hugging Face `evaluate` and `calc_perplexity` both returned `50107.171875` for `sshleifer/tiny-gpt2`; absolute delta was `0.0`.
- A uniform analytical model returned perplexity `10.0`; scalar total likelihood correctly used `N-1` causal predictions.
- A 4096-token retrieval target tokenized to 4119 tokens and preserved fact/question linkage.
- Identical fake models returned composite delta `0.0` and severity `excellent`.
- `get_severity_tier(0.50)` returned `poor` with `requires_confirmation=True`.
- `get_severity_tier(0.61)` returned `critical` with `quarantined=True`.

### Deviations from the roadmap

- Golden prompts use independent perplexity deltas because the validation contract stores pass/fail but does not require free-form generation.

### Needs manual verification

- Run the suite against an actual source model and its quantized artifact on hardware capable of loading both.

### Questions asked and answers

- The user conditionally approved the perplexity work after mathematical verification. The implementation was checked against Hugging Face and matched exactly.
## 2026-06-27 — Phase 8: Telemetry Pipeline

### Tasks completed

- 8.1 Added SQLAlchemy SQLite tables for jobs and validation summaries only.
- 8.2 Added shared-pool, fire-and-forget Redis hash writes.
- 8.3 Added an independent 10-second Redis aggregator with disabled, Prometheus, and PostgreSQL sink modes.
- 8.4 Added all five architecture metrics with warning, critical, and emergency policies.
- 8.5 Added a no-I/O, per-tick warning evaluator.

### Files created or modified

- `telemetry/db.py` — metadata database and job/validation operations.
- `telemetry/redis_pipeline.py` — non-blocking telemetry writes.
- `telemetry/aggregator.py` — independent batch collection and export.
- `telemetry/warnings.py` — in-memory threshold evaluation.
- `config/telemetry_thresholds.json` — five-metric warning policy.
- `tests/test_telemetry.py` — database, latency, decoupling, schema, and alert tests.
- `README.md`, `docs/ARCHITECTURE.md`, and `docs/BUILD_LOG.md` — operating guidance and architecture record.

### Verification commands and actual results

- SQLite first-run creation produced exactly `jobs` and `validation_results`; insertion and lookup returned the expected job.
- A Redis write backed by a deliberately 100 ms client returned its task in `0.0563 ms`.
- The slow-sink regression confirmed Redis writes complete while aggregation remains pending.
- Threshold JSON contains exactly five metrics and all three levels per metric.
- `evaluate_tick({"vram_pct": 98})` returned one `emergency` alert with action `abort`.
- `python -m pytest tests -q` returned `10 passed in 1.61s`.

### Deviations from the roadmap

- CPU emergency is explicitly disabled because Architecture §4.4 defines it as N/A; the emergency level remains present to satisfy the uniform policy schema.
- Following the distribution clarification, durable aggregate export is disabled by default. Prometheus and PostgreSQL are explicit team/server modes.
- Redis is a bundled local background process for the Enterprise Fat Binary, not a remote service dependency.

### Needs manual verification

- Exercise a production PostgreSQL sink under real server latency and failure conditions.

### Post-phase contract correction

- Re-audited Task 2.1 against the exact §1.2 tables and made `auth` a required `AuthBlock`.
- Added the specified top-level `gpu_uuids` field to `HardwareProfile`.
- Corrected `ModelMetaProfile` to expose `num_shards`, `total_weight_bytes`, and `quant_bits`.
- Corrected validation deltas to `PPL_quantized - PPL_original` rather than a relative ratio.
- Added exact field-name, required-auth, payload-profile, absolute-delta, and standalone-sink regression tests.

### Questions asked and answers

None.
## 2026-06-27 — Phase 9: Interface Shells

### Tasks completed

- 9.1 Added CLI `run`, `status`, and reserved `purge` commands using the Orchestrator contract.
- 9.2 Added an authenticated local FastAPI gateway with SSE submission and polling status.
- 9.3 Added a tkinter GUI with model/mode inputs, live state/telemetry, and a non-blocking worker thread.
- 9.4 Added a Kaggle/Colab adapter with secret loading, nested-loop support, progress, and displayable results.

### Files created or modified

- `cli/main.py` — authenticated terminal serializer and progress consumer.
- `cluster/api_server.py` — authentication middleware, SSE endpoint, and job status.
- `ui/app.py` — desktop interface with isolated async execution.
- `notebooks/adapter.py` — notebook detection, secrets, streaming, and result object.
- `tests/test_interfaces.py` — envelope, authentication, SSE, isolation, and notebook tests.
- `requirements.txt` — explicit interface runtime dependencies.
- `README.md`, `docs/ARCHITECTURE.md`, and `docs/BUILD_LOG.md` — interface operating record.

### Verification commands and actual results

- CLI help exposed `run`, `status`, and `purge`; a test key produced a strict authenticated CLI envelope.
- Missing API credentials returned HTTP 401 JSON before envelope processing.
- A valid simulated job returned an SSE `complete` event.
- GUI and notebook builders produced `gui`/JWT and `kaggle`/API-key envelopes respectively.
- AST inspection found no `engines` import in any interface module.
- A simulated notebook async stream completed and returned a displayable result.
- `python -m pytest tests -q` returned `21 passed`.

### Deviations from the roadmap

- The purge command is present but deliberately refuses execution until the five-phase purge controller is implemented in Phase 12.
- The local FastAPI app is transport-protocol agnostic; HTTP/2 and TLS termination are packaging/deployment configuration.

### Needs manual verification

- Launch the tkinter window on a graphical Windows session and observe a live job without UI freezing.
- Run the adapter in hosted Kaggle and Colab runtimes with notebook secrets.
- Verify TLS-terminated HTTP/2 only for optional team/server API deployment.

### Questions asked and answers

- The user clarified that the Enterprise Fat Binary is the primary deployment and supplied the exact §1.2 contract rules. Phase 9 uses those corrected contracts.
## 2026-06-27 — Phase 10: Prompt and Persona Infrastructure

### Tasks completed

- 10.1 Added model-family system-prompt formatting for Llama 3, Mistral, ChatML, Phi-3, Gemma, and unknown families.
- 10.2 Added the three required immutable persona presets.
- 10.3 Added exact native `/tokenize` calls with Hugging Face tokenizer plus 5% offline fallback.
- 10.4 Added online/offline context budgeting and structured overflow errors below 256 tokens.

### Files created or modified

- `core/prompt_controller.py` — native model-family prompt wrappers.
- `config/persona_presets.json` — three built-in non-deletable personas.
- `core/accelerator.py` — native/offline counting and context budgeting.
- `tests/test_prompt_controller.py` — prompt, persona, HTTP tokenizer, reserve, and overflow tests.
- `README.md`, `docs/ARCHITECTURE.md`, and `docs/BUILD_LOG.md` — Phase 10 operating record.

### Verification commands and actual results

- Llama 3 formatting matched the full header and end-of-turn token sequence.
- All five named family paths and the raw unknown-family fallback passed.
- Persona JSON contains exactly three required IDs with `deletable=false`.
- A real local aiohttp `/tokenize` endpoint returned exact count `7`.
- A 100-token offline estimate returned `105`.
- §6.3 budgeting returned `3796` online and `3781` offline for max 4096, system 200, and history 100.
- An under-256 budget returned `context_overflow_error` with its full token breakdown.
- `python -m pytest tests -q` returned `27 passed`.

### Deviations from the roadmap

- The Task 10.4 Done When example states `3746` online and approximately `3703` offline, but §6.3's explicit formula yields `3796` and `3781` for the supplied inputs. The implementation follows the authoritative architecture formula and does not invent an undocumented deduction.

### Needs manual verification

- Compare token counts against each packaged backend after its native endpoint is wired into the final executable.

### Questions asked and answers

None.
## 2026-06-28 — Trusted Local Authentication Bootstrap

### Changes

- Added automatic private local credentials for direct CLI, Kaggle, and Colab use.
- Preserved the mandatory §1.2 `AuthBlock` and authentication middleware path.
- Kept FastAPI/server authentication explicit and unchanged.
- Added `requirements-notebook.txt` without AutoGPTQ, AutoAWQ, or a forced Torch downgrade for Python 3.12 notebook runtimes.

### Verification

- Local credentials are reused, validate successfully, and never appear as plaintext in `credentials.json`.
- CLI and notebook envelope tests require no environment or notebook secret.
- `python -m pytest tests -q` returned `28 passed`.
## 2026-06-28 — Production Model-Flow Repair

### Changes

- Added conservative parameter-count planning when Hub metadata omits the count.
- Added format-aware GGUF selection, storage checks, sandboxed download, and GGUF v2/v3 header parsing.
- Connected repository acquisition to Orchestrator execution so GGUF workers receive a real local model path.
- Forced pre-quantized GGUF repositories through llama.cpp instead of hardware-only AWQ/vLLM routing.
- Added early actionable rejection for non-GGUF repositories routed to llama.cpp.

### Verification

- Live `sshleifer/tiny-gpt2` inspection now produces a conservative planning count instead of `None`.
- Live `ggml-org/tiny-llamas` resolution selected and downloaded `stories15M-q4_0.gguf` (19,077,344 bytes).
- The real GGUF header returned 6 layers, hidden size 288, 6 attention heads, context 128, and vocabulary 32,000.
- Artifact unit tests cover target selection, projection exclusion, incompatibility rejection, metadata parsing, and parameter estimation.
- A simulated full Orchestrator lifecycle proved the acquired GGUF path and source quantization format reach the worker and complete teardown.

### Remaining production gate

- This Windows host has no CMake/compiler toolchain, so genuine llama.cpp inference was not claimed locally.
- Kaggle/Colab must build llama.cpp CUDA and run the documented tiny GGUF smoke test.
- Phase 7 perplexity validation is not yet wired to the live GGUF backend; successful inference alone does not close production validation.

### Reality-aligned architecture correction

- Added `docs/ARCHITECTURE_V3.1_REALITY_DRAFT.md` and its DOCX rendering.
- Documented the missing operation semantics and source-versus-candidate distinction in v3.0.
- Defined operation-specific state paths and a fail-closed production acceptance gate.
- Tightened the current Orchestrator so a backend-specific output dictionary can no longer masquerade as `ValidationResult`; artifact delivery is blocked unless strict original-versus-quantized validation is available.
## 2026-06-28 — v3.1 Operation Contract Migration

### Changes

- Bumped strict wire contracts from schema 3.0 to 3.1.
- Replaced ambiguous `model_source` with required `operation`, `source_model`, `candidate_artifact`, `target`, and `validation_policy` fields.
- Added operation-specific validation: quantize requires a target; validate requires a candidate; required golden validation cannot be empty.
- Added distinct model-inspection and inference progress events.
- Added direct `inspect → teardown` and `infer → teardown` state paths.
- Kept quantization on the mandatory validation path.
- Migrated CLI, API payload handling, GUI, notebook adapter, Orchestrator, and tests together.

### Verification

- v3.0 event literals are rejected by the strict v3.1 contracts.
- Invalid quantize and validate envelopes fail schema construction.
- All automated tests pass after the migration.
## 2026-06-28 — Genuine Reference/Candidate Validation

### Changes

- Added lazy Transformers and subprocess-isolated llama-perplexity evaluators.
- Added async policy-selected logic, retrieval, code, and golden-prompt validation.
- Renormalized weights when a validation policy selects fewer than three domains.
- Connected v3.1 `validate` to reference/candidate resolution, strict `ValidationResult`, evaluator cleanup, SQLite job metadata, and validation-summary persistence.
- Critical validation remains fail-closed and enters the error/teardown path.

### Verification

- Fixed evaluators prove absolute PPL deltas and weighted severity behavior.
- Golden prompts remain separate from the composite.
- llama-perplexity output parsing accepts its final PPL line.
- An Orchestrator integration test persists a strict three-domain result and finishes through teardown.
