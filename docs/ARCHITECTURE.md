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
|   `-- __init__.py
|-- docs/
|   |-- __init__.py
|   |-- ARCHITECTURE.md
|   `-- BUILD_LOG.md
|-- engines/
|   `-- __init__.py
|-- notebooks/
|   `-- __init__.py
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

`core/` will contain shared schemas, authentication, hardware and model
inspection, strategy planning, orchestration, and prompt control. It currently
contains only its package marker.

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

No runtime data contracts are defined in Phase 1. `JobEnvelope`,
`ProgressEvent`, `AuthBlock`, `HardwareProfile`, `StrategyConfig`,
`ErrorEnvelope`, `TeardownComplete`, and model metadata contracts are scheduled
for Phase 2.

## Environment variables

No environment variables are required by the Phase 1 code. Later phases will
introduce authentication, cache, telemetry, database, and llama.cpp binary
variables.

## Known limitations

- The configuration files are empty skeletons.
- The import-isolation test is intentionally a placeholder until Phase 14.
- No schemas, authentication, profiler, model inspector, orchestrator, workers,
  interfaces, telemetry pipeline, cluster implementation, or packaging logic
  exists yet.
- The installed Torch build is CPU-only and reports CUDA unavailable.

## How to run this so far

From the repository root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import fastapi, redis, jwt"
python -c "import torch; print(torch.cuda.is_available())"
python -m pytest tests/test_import_isolation.py
```
