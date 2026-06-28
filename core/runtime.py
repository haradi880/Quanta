"""Resolve release-bundled native tools without runtime downloads."""

from __future__ import annotations

import os
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
        key = next(
            (
                name
                for name, filename in NATIVE_FILENAMES.items()
                if path.name.lower() == filename.lower()
            ),
            stem,
        )
        variable = NATIVE_ENV.get(key)
        if variable and variable not in os.environ:
            os.environ[variable] = str(path.resolve())
            configured[variable] = os.environ[variable]
    return configured
