"""Offline diagnostics for a packaged standalone runtime."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis
from core.runtime import NATIVE_ENV, configure_native_runtime
from telemetry.redis_manager import LocalRedisManager


async def _probe_process(
    path: str,
    *arguments: str,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
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
    text = output.decode("utf-8", errors="replace").strip()
    if process.returncode not in allowed_returncodes or not text:
        raise RuntimeError(
            f"native probe failed ({process.returncode}): {Path(path).name}"
        )
    return {
        "path": str(Path(path).resolve()),
        "version_output": text[:500],
    }


async def _verify_resp_operations(url: str) -> dict[str, Any]:
    client = Redis.from_url(url, decode_responses=True)
    key = f"haradibots:doctor:{uuid4()}"
    expected = {"gpu": "0", "status": "healthy"}
    try:
        ping = await client.ping()
        hset_count = await client.hset(key, mapping=expected)
        values = await client.hgetall(key)
        _, keys = await client.scan(cursor=0, match=key, count=100)
        scan_match = key in keys
    finally:
        await client.delete(key)
        await client.aclose()
    if ping is not True or values != expected or not scan_match:
        raise RuntimeError("Garnet RESP compatibility probe failed")
    return {
        "PING": "PONG",
        "HSET": hset_count,
        "HGETALL": values,
        "SCAN": keys,
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
    probes["llama-completion"] = await _probe_process(
        resolved["llama-completion"],
        "--version",
    )
    probes["llama-quantize"] = await _probe_process(
        resolved["llama-quantize"],
        "--help",
        allowed_returncodes=(0, 1),
    )
    probes["llama-perplexity"] = await _probe_process(
        resolved["llama-perplexity"],
        "--version",
    )

    manager = LocalRedisManager(binary_path=resolved["garnet-server"], port=0)
    # Reserve an ephemeral loopback port briefly, then hand it to the process.
    import socket

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        manager.port = reservation.getsockname()[1]
    url = await manager.start(timeout_seconds=15)
    try:
        redis_ok = await manager._reachable()
        resp_checks = await _verify_resp_operations(url)
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
        "resp_checks": resp_checks,
    }
