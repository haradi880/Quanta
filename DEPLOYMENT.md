# HaradiBots Deployment

HaradiBots has two deployment shapes. The Enterprise Fat Binary is the primary
single-machine product. The container image is for authenticated API/server
use. A release is not valid unless the native bundle verifier and offline smoke
test pass.

## Enterprise Fat Binary

1. Place release-pinned native tools in `build/vendor/`:
   `llama-cli`, `llama-quantize`, `llama-perplexity`,
   `convert_hf_to_gguf.py`, and `redis-server`.
2. Verify the offline payload:

   ```powershell
   python build/verify_bundle.py
   ```

3. Install the release build tool and create the one-dir distribution:

   ```powershell
   python -m pip install pyinstaller
   pyinstaller --clean --noconfirm build/fat_binary.spec
   ```

4. Copy `dist/HaradiBots/` to a clean machine with no Python installation.
5. Disconnect networking and run:

   ```powershell
   .\HaradiBots.exe --help
   ```

6. Submit a local GGUF inference job and verify no network call, import error,
   or surviving worker process.

The binary automatically resolves bundled native tools. Runtime downloads are
forbidden. Do not publish a release if `build/verify_bundle.py` fails.

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
