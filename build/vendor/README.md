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
  }
}
```

The release owner must confirm redistribution rights and preserve the upstream
license notices. The project deliberately does not fetch mutable `latest` URLs.
