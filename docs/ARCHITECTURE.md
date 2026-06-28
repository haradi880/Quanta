# Architecture

## Current structure

```text
.
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- build/
|   `-- __init__.py
|-- cli/
|   `-- __init__.py
|-- cluster/
|   `-- __init__.py
|-- config/
|   |-- __init__.py
|   |-- decision_matrix.json
|   |-- persona_presets.json
|   |-- telemetry_thresholds.json
|   `-- validation_suite.json
|-- core/
|   |-- __init__.py
|   |-- accelerator.py
|   |-- auth_middleware.py
|   |-- hf_inspector.py
|   |-- orchestrator.py
|   |-- profiler.py
|   `-- schemas.py
|-- docs/
|   |-- __init__.py
|   |-- ARCHITECTURE.md
|   `-- BUILD_LOG.md
|-- engines/
|   |-- __init__.py
|   |-- awq_worker.py
|   |-- base_worker.py
|   |-- exl2_worker.py
|   |-- gguf_worker.py
|   `-- vllm_worker.py
|-- notebooks/
|   `-- __init__.py
|-- scripts/
|   `-- gen_credentials.py
|-- telemetry/
|   `-- __init__.py
|-- tests/
|   |-- __init__.py
|   `-- test_import_isolation.py
`-- ui/
    `-- __init__.py
```

Local metadata and generated directories such as `.git`, `.venv`, `.pytest_cache`,
`.agents`, and `.codex` are omitted.

## Module responsibilities

`core/` contains the strict v3.1 wire schemas, authentication boundary,
hardware profiler, asynchronous Hugging Face inspector, strategy planner, and
orchestrator. The profiler gathers NVIDIA telemetry when NVML is available,
otherwise produces a topology-aware CPU profile; it also owns the thread plan,
three memory formulae, and hardware decision-matrix lookup. The inspector uses
only Hub metadata and small JSON configuration files to inventory files,
classify attention, detect prior quantization, and identify model family
before any weight download. The accelerator translates matrix policies into
concrete parallelism/layer settings and validates manual overrides using the
full conservative VRAM formula. The orchestrator is the sole interface entry
point, authenticates before entering its FSM, owns worker handles, and always
harvests registered process trees before returning to IDLE. Prompt control
follows in a later phase.

`engines/` contains the execution worker abstraction and GGUF, AWQ, EXL2, and
vLLM workers. Modules import no backend libraries at load time. GGUF is always
an isolated subprocess; AWQ loads AutoAWQ on instantiation; EXL2 runs the
official converter and loads ExLlamaV2 only for inference; vLLM passes a
concrete tensor-parallel degree and obtains exact token counts from `/tokenize`.

`cluster/` contains the authenticated FastAPI/SSE gateway, mTLS node-health
probe, and optional Ray, SLURM, and Kubernetes adapters. These adapters
provision jobs and report scheduler readiness; distributed model execution and
artifact return are not yet integrated and must not be advertised as complete.

`cli/` serializes terminal input into authenticated job envelopes and streams
Orchestrator events with rich progress. It never calls execution engines.
Trusted direct CLI and notebook interfaces bootstrap an internal local
credential automatically. This preserves the mandatory `AuthBlock` and
authentication gate without imposing API-key setup on single-machine users.
The network API boundary never uses this bootstrap and continues to require an
explicit caller credential.

`ui/` provides a tkinter desktop interface. It runs the Orchestrator stream on
a dedicated worker thread and drains events on the GUI main loop.

`notebooks/` provides Kaggle and Colab adapters with notebook-secret lookup,
nested event-loop support, and streaming progress display.

`telemetry/` separates persistent SQLite job metadata from the RESP telemetry
hot path. In the primary Enterprise Fat Binary deployment, Microsoft Garnet is
a bundled local background process, not a cloud dependency. RESP writes are scheduled
without awaiting network I/O. The independent 10-second aggregator has durable
export disabled by default for v1; Prometheus or PostgreSQL can be explicitly
enabled for team/server deployments. Threshold policies are loaded once and
evaluated in memory on every tick. SQLite remains local-only job metadata and
never stores high-frequency telemetry ticks.

`config/` holds validated runtime policies, including the validation suite and
five-metric telemetry threshold table.

`build/` contains certificate generation, the non-root CUDA container,
deterministic llama.cpp/Garnet vendor preparation, the fail-closed SHA-256
bundle verifier, and the one-directory PyInstaller build.

`tests/` contains unit, contract, packaging, cluster, lifecycle, security, and
native integration checks. Import isolation is enforced by an AST linter.

`docs/` contains the verified phase build record and this standing architecture
reference.

## Data contracts and schemas

All contracts live in `core/schemas.py`, use Pydantic v2 strict mode, and reject
undeclared fields.

- `JobEnvelope` is the inbound v3.1 contract. It carries `schema_version`,
  `job_id`, `auth`, `interface`, `mode`, `model_source`, optional hardware,
  quantization and cluster overrides, optional validation prompts,
  `system_prompt`, the deprecated display refresh hint, and callback channels.
- `AuthBlock` permits exactly one of an API key or JWT.
- `ProgressEvent` is the outbound stream contract with the v3.1 event type
  vocabulary, UTC timestamp, event payload, and latest telemetry snapshot.
- `HardwareProfile`, `GPUProfile`, and `CPUProfile` describe the hardware
  inventory consumed by planning. `core/profiler.py:snapshot()` now produces
  and validates this contract.
- `StrategyConfig` carries format, GPU layers, backend, parallelism degrees,
  and any safety warning.
- `ErrorEnvelope` and `TeardownComplete` define structured failure and process
  harvesting results.
- `ModelMetaProfile` defines Hugging Face inspection output, including GQA
  metadata, exact SafeTensor parameter count, and pre-quantization detection.
  `core/hf_inspector.py:inspect_repo` now produces and validates this contract.
- `ValidationResult` and its nested result models define per-domain,
  composite, severity, quarantine, confirmation, and golden-prompt results.

Phase 7 validation is implemented in `core/accelerator.py`. It uses causal
perplexity `exp(-mean(log p(x_i)))`, with `N-1` predicted tokens for a sequence
of length `N`. Domain deltas are the absolute difference
`PPL_quantized - PPL_original`. The default composite weights are logic `0.30`, retrieval
`0.35`, and code `0.35`. Severity boundaries are excellent through `0.05`,
good through `0.15`, moderate through `0.35`, poor through `0.60`, and critical
above `0.60`. Poor requires confirmation and critical is quarantined.

## Deployment Mode

The primary distribution is a single-machine Enterprise Fat Binary / executable
download. Python dependencies, CPU and CUDA 12.4 llama.cpp runtimes, and
Microsoft Garnet are bundled at build time. Runtime discovery selects the CUDA
tools only when an NVIDIA driver reports a GPU, otherwise it uses the CPU
tools; an explicit operator environment override remains authoritative.
Garnet is native Windows, uses a self-contained .NET runtime, and implements
the RESP protocol used by Redis clients. It is explicitly “close-enough,” not
100%-identical to Redis. The standalone telemetry path currently relies only
on `PING`, `HSET`, `HGETALL`, and `SCAN`; every future telemetry command must
be re-checked against Garnet’s API compatibility table before support is
assumed. PostgreSQL, Ray, SLURM, Kubernetes, and remote Prometheus are optional
team/server capabilities and are never prerequisites for standalone operation
or local Done When checks.

## Prompt and context control

`core/prompt_controller.py` applies model-family special tokens without
embedding interface logic. `core/accelerator.py` treats a live backend's
`/tokenize` response as authoritative. Offline estimation uses the model's
Hugging Face tokenizer with a 5% reserve. Context budgeting follows §6.3:
online budgets deduct system and history tokens exactly; offline budgets also
deduct 5% of their current total sequence length. Fewer than 256 available
generation tokens produces a structured breakdown instead of inference.

## Model artifact lifecycle

Repository inspection and execution are joined by `core/artifacts.py`.
Hardware planning may use a conservative weight-size parameter estimate when
the Hub omits `parameter_count`. GGUF repositories are resolved to one
non-projection model file compatible with the target format, checked against
free storage with a 10% reserve, downloaded only under
`$HARADIBOTS_CACHE_ROOT/models`, and parsed directly for architecture, layer,
attention, vocabulary, and context metadata. Pre-quantized GGUF forces the
llama.cpp backend instead of being routed to AWQ/vLLM based only on hardware.
A non-GGUF repository selected for llama.cpp is rejected before worker launch
with an actionable compatibility error.

This lifecycle is required before the optional cluster layer. CPU inference,
F32-to-Q4 quantization, and all-domain reference/candidate validation have
been exercised with the pinned bundled llama.cpp tools. CUDA and clean-machine
offline acceptance remain separate release gates.

## Environment variables

- `HARADIBOTS_JWT_SECRET` is required when validating JWT credentials. It must
  be a private, randomly generated signing secret and must not be committed.
- `.env` currently stores `DEV_API_KEY` for local development only. The file is
  ignored, and runtime interface wiring for it is scheduled for a later phase.
- API-key validation reads hashed records from ignored
  `config/credentials.json`.
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is optional and used only to inspect
  files in a gated/private Hugging Face repository for which the caller has
  access.
- `HARADIBOTS_LLAMA_BIN` is required for GGUF execution and identifies the
  compiled llama.cpp executable.
- `HARADIBOTS_EXL2_CONVERT_SCRIPT` is required for EXL2 conversion and
  identifies the official `convert.py`.
- `HARADIBOTS_CACHE_ROOT` optionally overrides the per-job work/output root;
  its default is `~/.haradibots/cache`.

## Verified scope and remaining release gates

The standalone Windows implementation has automated coverage for contracts,
authentication, profiling, inspection, planning, local execution, validation,
telemetry, degradation, teardown, purge, and packaging. The bundled Garnet
runtime has passed real `PING`, `HSET`, `HGETALL`, `SCAN`, and owned-process
shutdown checks. The pinned llama.cpp runtime has passed real CPU inference,
F32-to-Q4 quantization, and three-domain plus golden-prompt validation.

The project is not yet labeled production-ready because this host cannot prove
CUDA inference or clean-machine offline execution. Those two checks require a
Windows NVIDIA test machine and a second clean Windows machine respectively.
Optional Ray/SLURM/Kubernetes adapters also remain foundation-only until real
schedulers execute model work and return artifacts. See `DEPLOYMENT.md` for
the exact release procedure and `docs/RELEASE_STATUS.md` for the evidence
matrix.

## How to run this so far

From the repository root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import fastapi, redis, jwt"
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest tests/test_import_isolation.py
```

Generate a development credential once:

```powershell
python scripts\gen_credentials.py
```

The generator refuses to overwrite an existing `DEV_API_KEY`. Import the Phase
2 contracts and authentication gate with:

```powershell
python -c "from core.schemas import JobEnvelope; from core.auth_middleware import authenticate"
```

Produce and validate a hardware snapshot:

```powershell
python -c "from core.profiler import snapshot; from core.schemas import HardwareProfile; print(HardwareProfile.model_validate(snapshot()))"
```

Evaluate a 4 GiB GPU and 7B model against the matrix:

```powershell
python -c "from core.profiler import select_strategy; print(select_strategy({'gpu_count': 1, 'gpus': [{'vram_free_bytes': 4 * 1024**3}]}, {'model_size_b': 7, 'num_layers': 32}))"
```

Inspect a model repository without downloading weights:

```powershell
python -c "import asyncio; from core.hf_inspector import inspect_repo; print(asyncio.run(inspect_repo('NousResearch/Meta-Llama-3-8B')))"
```

Inspect the FSM transition guard:

```powershell
python -c "from core.orchestrator import JobState, transition; print(transition(JobState.IDLE, 'job_received'))"
```

Build a concrete automatic strategy:

```powershell
python -c "from core.accelerator import select_strategy; print(select_strategy({'gpu_count': 1, 'gpus': [{'vram_free_bytes': 4 * 1024**3}]}, {'parameter_count': 7_000_000_000, 'num_layers': 32, 'num_attention_heads': 32, 'num_key_value_heads': 8, 'hidden_size': 4096, 'max_position_embeddings': 8192}))"
```

Confirm worker modules preserve the lazy-import boundary:

```powershell
python -c "import sys; before=set(sys.modules); import engines.gguf_worker, engines.awq_worker, engines.exl2_worker, engines.vllm_worker; print((set(sys.modules)-before) & {'torch','transformers','awq','exllamav2','vllm','llama_cpp'})"
```

Check a conversion before dispatch:

```powershell
python -c "from core.accelerator import check_overcompilation; print(check_overcompilation('Q4_K_M', 'Q3_K_M'))"
```
