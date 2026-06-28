"""Offline diagnostics for a packaged standalone runtime."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from core.runtime import NATIVE_ENV, configure_native_runtime
from telemetry.redis_manager import LocalRedisManager


async def _probe_process(path: str, *arguments: str) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        path,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=15)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"native probe timed out: {Path(path).name}") from None
    if process.returncode != 0:
        raise RuntimeError(
            f"native probe failed ({process.returncode}): {Path(path).name}"
        )
    return {
        "path": str(Path(path).resolve()),
        "version_output": output.decode("utf-8", errors="replace").strip()[:500],
    }


async def run_offline_doctor() -> dict[str, Any]:
    """Verify native tools and own a complete local Redis start/stop cycle."""

    configure_native_runtime()
    resolved: dict[str, str] = {}
    for name, variable in NATIVE_ENV.items():
        path = os.environ.get(variable)
        if not path or not Path(path).is_file():
            raise RuntimeError(f"packaged native asset is unavailable: {name}")
        resolved[name] = path

    converter = Path(resolved["convert_hf_to_gguf.py"])
    for dependency in ("conversion", "gguf-py"):
        if not (converter.parent / dependency).is_dir():
            raise RuntimeError(f"packaged converter dependency is missing: {dependency}")

    probes = {}
    for name in ("llama-cli", "llama-quantize", "llama-perplexity"):
        probes[name] = await _probe_process(resolved[name], "--version")

    manager = LocalRedisManager(binary_path=resolved["redis-server"], port=0)
    # Reserve an ephemeral loopback port briefly, then hand it to the process.
    import socket

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        manager.port = reservation.getsockname()[1]
    url = await manager.start(timeout_seconds=15)
    try:
        redis_ok = await manager._reachable()
    finally:
        await manager.stop()
    if not redis_ok or manager.process is not None:
        raise RuntimeError("local Redis lifecycle probe failed")

    return {
        "status": "healthy",
        "offline_native_assets": probes,
        "converter": str(converter.resolve()),
        "redis_url": url,
        "redis_stopped": True,
    }
