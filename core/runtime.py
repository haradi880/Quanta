"""Resolve release-bundled native tools without runtime downloads."""

from __future__ import annotations

import os
import sys
from pathlib import Path


NATIVE_ENV = {
    "llama-cli": "HARADIBOTS_LLAMA_BIN",
    "llama-quantize": "HARADIBOTS_LLAMA_QUANTIZE_BIN",
    "llama-perplexity": "HARADIBOTS_LLAMA_PERPLEXITY_BIN",
    "redis-server": "HARADIBOTS_REDIS_BIN",
    "convert_hf_to_gguf.py": "HARADIBOTS_GGUF_CONVERT_SCRIPT",
}


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
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        key = path.name if path.name == "convert_hf_to_gguf.py" else stem
        variable = NATIVE_ENV.get(key)
        if variable and variable not in os.environ:
            os.environ[variable] = str(path.resolve())
            configured[variable] = os.environ[variable]
    return configured
