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

`core/` contains the strict v3.0 wire schemas, authentication boundary,
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

`cluster/` will contain the API gateway, node health logic, mTLS support, and
Ray, SLURM, and Kubernetes adapters. It currently contains only its package
marker.

`cli/` will serialize CLI input into job envelopes and stream orchestrator
events. It must not call execution engines directly. It currently contains only
its package marker.

`ui/` will provide the desktop interface and communicate only through the
orchestrator contract. It currently contains only its package marker.

`notebooks/` will provide Kaggle and Colab adapters that serialize requests and
display progress without importing execution tiers directly. It currently
contains only its package marker.

`telemetry/` will separate persistent job metadata from the Redis telemetry hot
path and background aggregation. It currently contains only its package marker.

`config/` holds JSON configuration read by runtime modules. Its four current
files are intentionally empty objects and will be populated by later phases.

`build/` will contain certificate, packaging, container, and deployment
artifacts. It currently contains only its package marker.

`tests/` contains automated checks. Its current import-isolation test is the
intentional Phase 1 `assert True` placeholder and will become an AST-based
linter in Phase 14.

`docs/` contains the verified phase build record and this standing architecture
reference.

## Data contracts and schemas

All contracts live in `core/schemas.py`, use Pydantic v2 strict mode, and reject
undeclared fields.

- `JobEnvelope` is the inbound v3.0 contract. It carries `schema_version`,
  `job_id`, `auth`, `interface`, `mode`, `model_source`, optional hardware,
  quantization and cluster overrides, optional validation prompts,
  `system_prompt`, the deprecated display refresh hint, and callback channels.
- `AuthBlock` permits exactly one of an API key or JWT.
- `ProgressEvent` is the outbound stream contract with the v3.0 event type
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
of length `N`. The default composite weights are logic `0.30`, retrieval
`0.35`, and code `0.35`. Severity boundaries are excellent through `0.05`,
good through `0.15`, moderate through `0.35`, poor through `0.60`, and critical
above `0.60`. Poor requires confirmation and critical is quarantined.

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

## Known limitations

- The configuration files are empty skeletons.
- The import-isolation test is intentionally a placeholder until Phase 14.
- No profiler, model inspector, orchestrator, workers, interfaces, telemetry
  pipeline, cluster implementation, or packaging logic exists yet.
- The installed Torch build is CPU-only and reports CUDA unavailable.
- JWT validation currently uses one environment-provided HS256 signing key;
  rotating key-set support is not yet implemented.
- NVIDIA telemetry has not been exercised on real NVIDIA hardware in this
  environment.
- Hybrid P/E-core classification has synthetic coverage but has not been
  exercised on a real hybrid CPU here.
- Full config inspection of Meta's original gated Llama repositories has not
  been verified with an authorized account token.
- Real backend subprocess harvesting and Ray actor termination remain
  integration-verification items for Phases 6 and 11.
- GGUF has not been run with a real llama.cpp binary/model in this environment.
- AutoAWQ imports, but its CUDA kernels and real quantization are unavailable
  on this CPU host.
- ExLlamaV2 and vLLM are not installed because they require
  platform/CUDA-specific builds; clean missing-backend behavior is verified.

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
