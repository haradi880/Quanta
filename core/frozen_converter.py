"""Private launcher for a bundled llama.cpp HF-to-GGUF converter."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_bundled_converter(arguments: list[str]) -> int:
    """Execute the pinned converter inside the frozen Python runtime."""

    if not arguments:
        raise SystemExit("bundled converter path is required")
    converter = Path(arguments[0]).resolve()
    if not converter.is_file() or converter.name != "convert_hf_to_gguf.py":
        raise SystemExit("bundled converter path is invalid")
    vendor_root = converter.parent
    gguf_python = vendor_root / "gguf-py"
    if not gguf_python.is_dir():
        raise SystemExit("bundled converter dependency gguf-py is missing")
    conversion = vendor_root / "conversion"
    if not conversion.is_dir():
        raise SystemExit("bundled converter dependency conversion is missing")
    sys.path.insert(0, str(vendor_root))
    sys.path.insert(0, str(gguf_python))
    sys.argv = [str(converter), *arguments[1:]]
    runpy.run_path(str(converter), run_name="__main__")
    return 0
