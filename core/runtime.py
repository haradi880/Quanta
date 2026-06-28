"""Resolve release-bundled native tools without runtime downloads."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


NATIVE_ENV = {
    "llama-completion": "HARADIBOTS_LLAMA_BIN",
    "llama-quantize": "HARADIBOTS_LLAMA_QUANTIZE_BIN",
    "llama-perplexity": "HARADIBOTS_LLAMA_PERPLEXITY_BIN",
    "garnet-server": "HARADIBOTS_GARNET_BIN",
    "convert_hf_to_gguf.py": "HARADIBOTS_GGUF_CONVERT_SCRIPT",
}
NATIVE_FILENAMES = {
    "llama-completion": "llama-completion.exe",
    "llama-quantize": "llama-quantize.exe",
    "llama-perplexity": "llama-perplexity.exe",
    "garnet-server": "GarnetServer.exe",
    "convert_hf_to_gguf.py": "convert_hf_to_gguf.py",
}


def _nvidia_driver_available() -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False


def configure_native_runtime(bundle_root: Path | None = None) -> dict[str, str]:
    root = bundle_root
    if root is None:
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            root = Path(frozen_root) / "vendor"
        else:
            root = Path(__file__).resolve().parents[1] / "build" / "vendor"
    configured: dict[str, str] = {}
    if not root.exists():
        return configured
    use_cuda = _nvidia_driver_available() and (root / "cuda").is_dir()
    for key, variable in NATIVE_ENV.items():
        if variable in os.environ:
            continue
        filename = NATIVE_FILENAMES[key]
        candidates = []
        if use_cuda and key in {
            "llama-completion",
            "llama-quantize",
            "llama-perplexity",
        }:
            candidates.append(root / "cuda" / filename)
        if key == "garnet-server":
            candidates.append(root / "garnet" / filename)
        candidates.append(root / filename)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            os.environ[variable] = str(path.resolve())
            configured[variable] = os.environ[variable]
    return configured
