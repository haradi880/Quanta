"""Fail a release build when required offline native assets are absent."""

from __future__ import annotations

import os
from pathlib import Path


REQUIRED_STEMS = {
    "llama-cli",
    "llama-quantize",
    "llama-perplexity",
    "redis-server",
}


def verify_vendor(vendor_root: Path) -> dict[str, str]:
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
    return {
        name: str(indexed[name].resolve())
        for name in sorted(REQUIRED_STEMS)
    } | {"convert_hf_to_gguf.py": str(converters[0].resolve())}


if __name__ == "__main__":
    root = Path(
        os.environ.get(
            "HARADIBOTS_VENDOR_ROOT",
            Path(__file__).resolve().parent / "vendor",
        )
    )
    verified = verify_vendor(root)
    print(f"Verified {len(verified)} required offline runtime assets.")
