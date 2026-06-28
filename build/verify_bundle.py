"""Fail a release build when required offline native assets are absent."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path


REQUIRED_STEMS = {
    "llama-cli",
    "llama-quantize",
    "llama-perplexity",
    "redis-server",
}
MANIFEST_NAME = "vendor-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_vendor(vendor_root: Path) -> dict[str, str]:
    manifest_path = vendor_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"offline bundle is incomplete; missing: {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = manifest["assets"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("offline vendor manifest is invalid") from exc
    if not isinstance(assets, dict):
        raise RuntimeError("offline vendor manifest is invalid")

    files = [path for path in vendor_root.rglob("*") if path.is_file()]
    indexed = {path.stem.lower(): path for path in files}
    missing = sorted(name for name in REQUIRED_STEMS if name not in indexed)
    converters = [path for path in files if path.name == "convert_hf_to_gguf.py"]
    if not converters:
        missing.append("convert_hf_to_gguf.py")
    if missing:
        raise RuntimeError(
            "offline bundle is incomplete; missing: " + ", ".join(missing)
        )
    verified = {
        name: str(indexed[name].resolve())
        for name in sorted(REQUIRED_STEMS)
    } | {"convert_hf_to_gguf.py": str(converters[0].resolve())}
    for name, resolved in verified.items():
        entry = assets.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise RuntimeError(f"offline vendor manifest has no checksum for: {name}")
        expected = entry["sha256"].lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise RuntimeError(f"offline vendor manifest checksum is invalid for: {name}")
        if _sha256(Path(resolved)) != expected:
            raise RuntimeError(f"offline vendor checksum mismatch for: {name}")
    return verified


if __name__ == "__main__":
    root = Path(
        os.environ.get(
            "HARADIBOTS_VENDOR_ROOT",
            Path(__file__).resolve().parent / "vendor",
        )
    )
    verified = verify_vendor(root)
    print(f"Verified {len(verified)} required offline runtime assets.")
