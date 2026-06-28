# Bundled native runtime

This directory is populated deterministically by:

```powershell
.\build\populate_windows_vendor.ps1 -Python .\.venv\Scripts\python.exe
```

The script builds and stages:

- pinned official llama.cpp Windows CPU executables;
- matching `convert_hf_to_gguf.py`, `conversion/`, and `gguf-py/`;
- Microsoft Garnet v1.1.10 as a self-contained native Windows x64 RESP server;
- Garnet’s MIT `LICENSE` and `NOTICE.md`;
- the Microsoft .NET Library License and .NET third-party notices.

`vendor-manifest.json` records provenance and SHA-256 for every file. The
release verifier rejects missing, modified, or unlisted payload files. Runtime
downloads are forbidden.

Garnet is deliberately treated as RESP-compatible, not identical to Redis.
Only `PING`, `HSET`, `HGETALL`, and `SCAN` are required by the current local
telemetry path and exercised by `HaradiBots doctor --json`. Re-check Microsoft’s
Garnet API compatibility table before adding another command.
