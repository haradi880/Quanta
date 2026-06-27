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
|   |-- auth_middleware.py
|   |-- profiler.py
|   `-- schemas.py
|-- docs/
|   |-- __init__.py
|   |-- ARCHITECTURE.md
|   `-- BUILD_LOG.md
|-- engines/
|   `-- __init__.py
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

`core/` contains the strict v3.0 wire schemas, authentication boundary, and
hardware profiler. The profiler gathers NVIDIA telemetry when NVML is
available, otherwise produces a topology-aware CPU profile; it also owns the
thread plan, three memory formulae, and hardware decision-matrix lookup.
Model inspection, orchestration, and prompt control follow in later phases.
Authentication supports hashed API keys and signed JWTs and stores the
verified identity in a request-scoped context variable.

`engines/` will contain the execution worker abstraction and the GGUF, AWQ,
EXL2, and vLLM workers. It currently contains only its package marker.

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
  metadata and pre-quantization detection.
- `ValidationResult` and its nested result models define per-domain,
  composite, severity, quarantine, confirmation, and golden-prompt results.

## Environment variables

- `HARADIBOTS_JWT_SECRET` is required when validating JWT credentials. It must
  be a private, randomly generated signing secret and must not be committed.
- `.env` currently stores `DEV_API_KEY` for local development only. The file is
  ignored, and runtime interface wiring for it is scheduled for a later phase.
- API-key validation reads hashed records from ignored
  `config/credentials.json`.

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
