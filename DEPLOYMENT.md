# HaradiBots Deployment

HaradiBots has two deployment shapes. The Enterprise Fat Binary is the primary
single-machine product. The container image is for authenticated API/server
use. A release is not valid unless the native bundle verifier and offline smoke
test pass.

## Enterprise Fat Binary

1. Run `build/populate_windows_vendor.ps1`. The script retrieves pinned
   llama.cpp sources and builds pinned Microsoft Garnet as a self-contained
   native Windows x64 RESP server. It preserves all required license/NOTICE
   files and writes a full SHA-256 inventory. See `build/vendor/README.md`.
2. Verify the offline payload:

   ```powershell
   python -m build.verify_bundle
   ```

3. Install the release build tool and create the one-dir distribution:

   ```powershell
   python -m pip install -r requirements-dev.txt
   .\build\build_windows.ps1 -Python python -Clean
   ```

4. Copy `dist/HaradiBots/` to a clean machine with no Python installation.
5. Disconnect networking and run:

   ```powershell
   .\HaradiBots.exe --help
   .\HaradiBots.exe doctor --json
   ```

6. The doctor must report all required llama.cpp tools, converter dependencies,
   and a successful owned Garnet
   `PING`/`HSET`/`HGETALL`/`SCAN`/stop cycle. Then submit a local GGUF inference
   job and verify no network call, import error, or surviving worker process.

The binary automatically resolves bundled native tools. Its private frozen
converter launcher executes the pinned converter and bundled `conversion/` and
`gguf-py/` sources inside the frozen Python runtime. Runtime downloads are
forbidden. Do not publish a release if `python -m build.verify_bundle` fails.

## Docker API

Build and run:

```bash
docker build -f build/Dockerfile -t haradibots:local .
docker run --rm --gpus all -p 8000:8000 \
  -v haradibots-cache:/var/lib/haradibots/cache \
  haradibots:local
curl http://127.0.0.1:8000/health
```

The health endpoint is public. Job endpoints require an explicit API key or
JWT. Supply credential configuration through a secret mount; never bake
`config/credentials.json` or `.env` into the image.

## Release gate

Before tagging a version:

- all CI jobs pass;
- coverage is at least 80%;
- native binaries are pinned and their checksums recorded;
- Fat Binary offline inference passes on clean Windows;
- container health and authenticated SSE submission pass;
- real reference-versus-candidate validation covers all three domains;
- teardown leaves no child processes;
- the security and dependency audits contain no unresolved critical finding.

Ray, SLURM, Kubernetes, and multi-node mTLS are optional cluster extensions and
are not prerequisites for the standalone Fat Binary.

The current evidence and the two outstanding standalone external checks are
tracked in `docs/RELEASE_STATUS.md`. Passing CI or building on the development
host does not substitute for the clean-machine offline step.
