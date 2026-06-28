"""Lifecycle owner for the bundled standalone Redis process."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any


class LocalRedisManager:
    def __init__(
        self,
        *,
        binary_path: str | None = None,
        host: str = "127.0.0.1",
        port: int = 6379,
    ) -> None:
        self.binary_path = binary_path or os.environ.get("HARADIBOTS_REDIS_BIN")
        self.host = host
        self.port = port
        self.process: asyncio.subprocess.Process | None = None
        self.owns_process = False

    def _resolve_binary(self) -> str:
        candidates = [
            self.binary_path,
            str(Path(sys.executable).resolve().parent / "redis-server.exe"),
            str(Path(sys.executable).resolve().parent / "redis-server"),
            shutil.which("redis-server"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return str(Path(candidate).resolve())
        raise FileNotFoundError(
            "bundled Redis binary was not found; set HARADIBOTS_REDIS_BIN"
        )

    async def _reachable(self) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=0.25,
            )
            writer.write(b"*1\r\n$4\r\nPING\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.readline(), timeout=0.25)
            writer.close()
            await writer.wait_closed()
            return response.startswith(b"+PONG")
        except (OSError, asyncio.TimeoutError):
            return False

    async def start(self, timeout_seconds: float = 5.0) -> str:
        if await self._reachable():
            os.environ["REDIS_URL"] = f"redis://{self.host}:{self.port}/0"
            return os.environ["REDIS_URL"]
        binary = self._resolve_binary()
        cache_root = Path(
            os.environ.get(
                "HARADIBOTS_CACHE_ROOT",
                str(Path.home() / ".haradibots" / "cache"),
            )
        ).expanduser()
        data_dir = cache_root / "redis"
        data_dir.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)
        self.process = await asyncio.create_subprocess_exec(
            binary,
            "--bind",
            self.host,
            "--protected-mode",
            "yes",
            "--port",
            str(self.port),
            "--dir",
            str(data_dir.resolve()),
            "--save",
            "",
            "--appendonly",
            "no",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.owns_process = True
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(
                    f"bundled Redis exited with code {self.process.returncode}"
                )
            if await self._reachable():
                os.environ["REDIS_URL"] = f"redis://{self.host}:{self.port}/0"
                return os.environ["REDIS_URL"]
            await asyncio.sleep(0.05)
        await self.stop()
        raise TimeoutError("bundled Redis did not become ready")

    async def stop(self) -> None:
        if not self.owns_process or self.process is None:
            return
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        self.owns_process = False


_LOCAL_REDIS = LocalRedisManager()


async def start_local_redis() -> str:
    return await _LOCAL_REDIS.start()


async def stop_local_redis() -> None:
    await _LOCAL_REDIS.stop()
