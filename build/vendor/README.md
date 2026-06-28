# Bundled native runtime

Place release-pinned native executables here before building:

- `llama-cli`
- `llama-quantize`
- `llama-perplexity`
- `convert_hf_to_gguf.py`
- `redis-server`

The release verifier rejects a Fat Binary build when any required executable is
missing. Runtime dependency downloads are forbidden.
