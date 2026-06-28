# Bundled native runtime

Place release-pinned native executables here before building:

- `llama-cli`
- `llama-quantize`
- `llama-perplexity`
- `convert_hf_to_gguf.py`
- `redis-server`

The release verifier rejects a Fat Binary build when any required executable is
missing or differs from its pinned SHA-256 digest. Runtime dependency downloads
are forbidden.

Create `vendor-manifest.json` beside this file:

```json
{
  "schema_version": "1",
  "assets": {
    "llama-cli": {"sha256": "<64 lowercase hex characters>", "source": "<release URL>", "license": "<SPDX identifier>"},
    "llama-quantize": {"sha256": "<64 lowercase hex characters>", "source": "<release URL>", "license": "<SPDX identifier>"},
    "llama-perplexity": {"sha256": "<64 lowercase hex characters>", "source": "<release URL>", "license": "<SPDX identifier>"},
    "convert_hf_to_gguf.py": {"sha256": "<64 lowercase hex characters>", "source": "<release URL>", "license": "<SPDX identifier>"},
    "redis-server": {"sha256": "<64 lowercase hex characters>", "source": "<release URL>", "license": "<SPDX identifier>"}
  },
  "files": {"relative/path": "<SHA-256 for every bundled payload file>"}
}
```

The release owner must confirm redistribution rights and preserve the upstream
license notices. The project deliberately does not fetch mutable `latest` URLs.

For Windows, `build/populate_windows_vendor.ps1` downloads the pinned official
llama.cpp CPU release and its matching converter source. Redis does not publish
a native Windows server; provide a licensed Redis-compatible executable and its
license explicitly:

```powershell
.\build\populate_windows_vendor.ps1 `
  -RedisBinary C:\licensed-runtime\redis-server.exe `
  -RedisRuntimeDirectory C:\licensed-runtime `
  -RedisSource https://vendor.example/release `
  -RedisLicense Proprietary `
  -RedisLicenseFile C:\licensed-runtime\LICENSE.txt `
  -RedisRedistributionApproved
```

The approval switch is an auditable release-owner assertion, not an automatic
license determination. Never use the abandoned Microsoft Redis Windows port in
a production release.
