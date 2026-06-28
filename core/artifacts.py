"""Format-aware model artifact selection and sandboxed acquisition."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download


class ArtifactCompatibilityError(RuntimeError):
    pass


_GGUF_SCALARS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _cache_root() -> Path:
    return Path(
        os.environ.get(
            "HARADIBOTS_CACHE_ROOT",
            str(Path.home() / ".haradibots" / "cache"),
        )
    ).expanduser()


def _normalized_format(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def select_gguf_file(
    file_manifest: dict[str, int],
    target_format: str,
) -> str:
    candidates = [
        name
        for name in file_manifest
        if name.lower().endswith(".gguf") and "mmproj" not in name.lower()
    ]
    if not candidates:
        raise ArtifactCompatibilityError(
            "selected llama.cpp backend requires a GGUF repository; "
            "choose a repository containing a .gguf model file"
        )
    target = _normalized_format(target_format.split("/")[0])
    matching = [
        name for name in candidates if target in _normalized_format(Path(name).stem)
    ]
    pool = matching or candidates
    return min(
        pool,
        key=lambda name: (
            file_manifest.get(name, 0) <= 0,
            file_manifest.get(name, 0),
            name,
        ),
    )


async def acquire_gguf_artifact(
    repo_id: str,
    model_meta: dict[str, Any],
    target_format: str,
    *,
    revision: str | None = None,
) -> str:
    """Download one compatible GGUF file into the isolated runtime cache."""

    filename = select_gguf_file(model_meta["file_manifest"], target_format)
    expected_size = int(model_meta["file_manifest"].get(filename, 0))
    destination = _cache_root() / "models" / repo_id.replace("/", "--")
    destination.mkdir(parents=True, exist_ok=True)
    if expected_size > 0:
        free_bytes = shutil.disk_usage(destination).free
        required_bytes = int(expected_size * 1.10)
        if free_bytes < required_bytes:
            raise OSError(
                f"insufficient storage for {filename}: need {required_bytes} "
                f"bytes including reserve, have {free_bytes}"
            )

    path = await asyncio.to_thread(
        hf_hub_download,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=destination,
    )
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise OSError(f"downloaded GGUF artifact is missing or empty: {resolved}")
    return str(resolved)


async def acquire_source_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
) -> str:
    """Acquire conversion inputs only, inside the fixed sandbox cache."""

    destination = _cache_root() / "sources" / repo_id.replace("/", "--")
    destination.mkdir(parents=True, exist_ok=True)
    path = await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "*.bin",
            "*.bin.index.json",
            "*.json",
            "tokenizer.model",
            "tokenizer.*",
            "*.txt",
        ],
    )
    resolved = Path(path).resolve()
    if not (resolved / "config.json").is_file():
        raise OSError("source snapshot is missing config.json")
    if not any(
        item.suffix.lower() in {".safetensors", ".bin"}
        for item in resolved.rglob("*")
        if item.is_file()
    ):
        raise OSError("source snapshot contains no supported full-precision weights")
    return str(resolved)


def inspect_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Read the GGUF metadata header without loading tensor data."""

    def read_exact(handle, size: int) -> bytes:
        value = handle.read(size)
        if len(value) != size:
            raise ValueError("truncated GGUF metadata")
        return value

    def read_string(handle) -> str:
        length = struct.unpack("<Q", read_exact(handle, 8))[0]
        if length > 64 * 1024 * 1024:
            raise ValueError("GGUF metadata string exceeds safety limit")
        return read_exact(handle, length).decode("utf-8", errors="replace")

    def read_value(handle, value_type: int) -> Any:
        if value_type in _GGUF_SCALARS:
            fmt = _GGUF_SCALARS[value_type]
            return struct.unpack(fmt, read_exact(handle, struct.calcsize(fmt)))[0]
        if value_type == 8:
            return read_string(handle)
        if value_type == 9:
            element_type = struct.unpack("<I", read_exact(handle, 4))[0]
            count = struct.unpack("<Q", read_exact(handle, 8))[0]
            if count > 10_000_000:
                raise ValueError("GGUF metadata array exceeds safety limit")
            if element_type in _GGUF_SCALARS:
                fmt = _GGUF_SCALARS[element_type]
                size = struct.calcsize(fmt)
                handle.seek(size * count, 1)
                return {"count": count}
            for _ in range(count):
                read_value(handle, element_type)
            return {"count": count}
        raise ValueError(f"unsupported GGUF metadata type {value_type}")

    with Path(path).open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError("file is not GGUF")
        version = struct.unpack("<I", read_exact(handle, 4))[0]
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        struct.unpack("<Q", read_exact(handle, 8))[0]  # tensor count
        metadata_count = struct.unpack("<Q", read_exact(handle, 8))[0]
        if metadata_count > 100_000:
            raise ValueError("GGUF metadata entry count exceeds safety limit")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = struct.unpack("<I", read_exact(handle, 4))[0]
            metadata[key] = read_value(handle, value_type)

    architecture = metadata.get("general.architecture")

    def architecture_value(suffix: str) -> Any:
        if isinstance(architecture, str):
            value = metadata.get(f"{architecture}.{suffix}")
            if value is not None:
                return value
        matches = [
            value
            for key, value in metadata.items()
            if key.endswith(f".{suffix}")
        ]
        return matches[0] if matches else None

    tokens = metadata.get("tokenizer.ggml.tokens")
    return {
        "model_family": architecture if isinstance(architecture, str) else None,
        "num_layers": architecture_value("block_count"),
        "hidden_size": architecture_value("embedding_length"),
        "num_attention_heads": architecture_value("attention.head_count"),
        "num_key_value_heads": architecture_value("attention.head_count_kv"),
        "max_position_embeddings": architecture_value("context_length"),
        "vocab_size": tokens.get("count") if isinstance(tokens, dict) else None,
    }
