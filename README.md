# HaradiBots

HaradiBots is an LLM quantization and profiling engine.

The repository is being built phase by phase from the HaradiBots v3.0 build roadmap.

## Development setup

HaradiBots requires Python 3.11 or newer. On Windows, create and activate a
virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For a CPU-only installation, install Torch from the PyTorch CPU index before
installing the complete requirements:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the Phase 1 test with:

```powershell
python -m pytest tests/test_import_isolation.py
```

## Local authentication setup

Generate a development API key once:

```powershell
python scripts\gen_credentials.py
```

The command writes the plaintext key only to the ignored `.env` file and its
SHA-256 hash only to the ignored `config/credentials.json` store. It does not
print the key. To exercise JWT authentication, set a secret of at least 32
random bytes in `HARADIBOTS_JWT_SECRET`; do not commit it.

## Hardware snapshot

Inspect the current machine without downloading a model:

```powershell
python -c "from core.profiler import snapshot; print(snapshot())"
```

On systems without an NVIDIA device or working NVML driver, the profiler
returns an empty GPU list and a complete CPU fallback profile. Strategy
selection reads `config/decision_matrix.json`.

## Hugging Face metadata inspection

Inspect repository metadata asynchronously without downloading model weights:

```powershell
python -c "import asyncio; from core.hf_inspector import inspect_repo; print(asyncio.run(inspect_repo('NousResearch/Meta-Llama-3-8B')))"
```

Set `HF_TOKEN` only when inspecting files in a gated or private repository for
which your Hugging Face account already has access. Do not commit the token.

## Orchestration

All interfaces will call the same asynchronous generator:

```python
from core.orchestrator import process_job

async for event in process_job(envelope):
    print(event.event_type, event.payload)
```

Authenticated jobs always pass through the `TEARDOWN` state before returning
to `IDLE`. Execution requires a registered worker; concrete workers are added
in Phase 6.

## Execution backends

Backend modules are safe to import without loading their ML libraries. The
selected worker loads its backend only when instantiated.

- GGUF uses only a llama.cpp subprocess. Set `HARADIBOTS_LLAMA_BIN` to the
  platform-appropriate `llama-cli` or compatible binary.
- AWQ uses AutoAWQ when the selected strategy requests it.
- EXL2 uses ExLlamaV2 and its official conversion script. Set
  `HARADIBOTS_EXL2_CONVERT_SCRIPT` to `convert.py`.
- vLLM creates `LLM(tensor_parallel_size=...)` and uses the running server's
  `/tokenize` endpoint for exact token counts.

EXL2 and vLLM require GPU/platform-specific installations and are intentionally
not installed by the generic CPU setup.

## Validation

Phase 7 compares original and quantized model perplexity across logic,
long-context retrieval, and code prompts:

```powershell
python -c "from core.accelerator import get_severity_tier; print(get_severity_tier(0.50))"
```

Golden prompts are scored separately and do not affect the weighted composite.
Poor results require confirmation; critical results are quarantined.

## Telemetry

Job metadata is stored under
`$HARADIBOTS_CACHE_ROOT/db/haradibots.sqlite3`. High-frequency metrics use the
local Redis process bundled with the Enterprise Fat Binary:

```python
from telemetry.redis_pipeline import write_tick

write_tick("job-id", "node-id", {"vram_pct": 72, "gpu_temp_c": 65})
```

`write_tick()` schedules Redis work and returns immediately. It must be called
from a running asyncio event loop. Durable aggregate export is disabled by
default. Team/server deployments may explicitly set
`HARADIBOTS_TELEMETRY_SINK=prometheus` or `postgresql`; PostgreSQL additionally
requires `POSTGRES_URL`.

## Interfaces

All interfaces construct the same authenticated version 3.0 `JobEnvelope` and
consume the same `ProgressEvent` stream:

```powershell
$env:HARADIBOTS_API_KEY = "<your local key>"
python -m cli.main run --model owner/model --mode auto
uvicorn cluster.api_server:app --host 127.0.0.1 --port 8000
python -m ui.app
```

Notebook users call `notebooks.adapter.run("owner/model")`. The desktop GUI
reads its session JWT from `HARADIBOTS_SESSION_JWT` or
`$HARADIBOTS_CACHE_ROOT/session.jwt`. No interface imports an execution engine.

CLI, Kaggle, and Colab direct calls automatically create a private local
credential under `$HARADIBOTS_CACHE_ROOT/auth/local.key`; users do not need to
create or supply an API key. The FastAPI gateway still requires explicit
header authentication. Notebook environments should install
`requirements-notebook.txt`, which excludes platform-specific compiled
backends and preserves the runtime-provided Torch build.

For a real Kaggle/Colab GGUF smoke test, build llama.cpp in the notebook,
set `HARADIBOTS_LLAMA_BIN` to its `llama-cli`, and use a GGUF repository such
as `ggml-org/tiny-llamas`. HaradiBots selects one compatible GGUF file,
downloads it into the sandbox cache, reads its header metadata, and passes the
resolved local file—not the repository name—to the worker.

## Prompt and context control

`core.prompt_controller.format_system_prompt()` formats system prompts for
Llama 3, Mistral, ChatML, Phi-3, and Gemma. The accelerator uses a running
backend's native `/tokenize` endpoint whenever possible. Offline estimates use
the supplied Hugging Face tokenizer with a 5% reserve, and context budgets
return a full `context_overflow_error` breakdown below 256 generation tokens.
