"""GGUF execution through an isolated llama.cpp subprocess."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from core.schemas import EventType, ProgressEvent
from engines.base_worker import BaseWorker


PERCENT_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")


class GGUFWorker(BaseWorker):
    """Manage llama-cli/llama-server as a subprocess, never a Python library."""

    def __init__(self, job_id: UUID, binary_path: str | None = None) -> None:
        super().__init__(job_id)
        self._import_backend()
        configured_path = binary_path or os.environ.get("HARADIBOTS_LLAMA_BIN")
        self.binary_path = configured_path
        self.process: asyncio.subprocess.Process | None = None
        self._last_strategy: dict[str, Any] | None = None

    @classmethod
    def _import_backend(cls) -> None:
        return None

    def _resolved_binary(self) -> str | None:
        if not self.binary_path:
            return None
        direct = Path(self.binary_path).expanduser()
        if direct.is_file():
            return str(direct.resolve())
        return shutil.which(self.binary_path)

    def _command(self, strategy_config: dict[str, Any]) -> list[str]:
        binary = self._resolved_binary()
        if binary is None:
            raise FileNotFoundError(
                "llama.cpp binary is not configured or does not exist; "
                "set HARADIBOTS_LLAMA_BIN"
            )
        model_path = strategy_config.get("model_path")
        if not isinstance(model_path, str) or not Path(model_path).is_file():
            raise FileNotFoundError("strategy_config.model_path must be a GGUF file")

        command = [binary, "-m", str(Path(model_path).resolve())]
        prompt = strategy_config.get("prompt")
        if isinstance(prompt, str):
            command.extend(["-p", prompt])
        if strategy_config.get("max_tokens") is not None:
            command.extend(["-n", str(int(strategy_config["max_tokens"]))])
        if strategy_config.get("gpu_layers") is not None:
            command.extend(["-ngl", str(int(strategy_config["gpu_layers"]))])
        if strategy_config.get("threads") is not None:
            command.extend(["-t", str(int(strategy_config["threads"]))])
        extra_args = strategy_config.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(argument, str) for argument in extra_args
        ):
            raise ValueError("strategy_config.extra_args must be a string array")
        command.extend(extra_args)
        return command

    async def execute(
        self,
        strategy_config: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        self._last_strategy = dict(strategy_config)
        try:
            command = self._command(strategy_config)
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            yield self._error_event("gguf", str(exc))
            return

        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {
                "backend": "gguf",
                "status": "launched",
                "pid": self.process.pid,
            },
        )

        if self.process.stdout is None:
            yield self._error_event("gguf", "subprocess stdout pipe was not created")
            return
        async for raw_line in self.process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            percentage_match = PERCENT_PATTERN.search(line)
            payload: dict[str, Any] = {
                "backend": "gguf",
                "status": "running",
                "message": line,
            }
            if percentage_match:
                payload["progress_pct"] = min(
                    max(float(percentage_match.group(1)), 0.0),
                    100.0,
                )
            yield self._event(EventType.QUANTIZATION_PROGRESS, payload)

        return_code = await self.process.wait()
        if return_code != 0:
            yield self._error_event(
                "gguf",
                f"llama.cpp subprocess exited with code {return_code}",
            )
            return
        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {"backend": "gguf", "status": "complete", "progress_pct": 100.0},
        )

    async def validate(
        self,
        prompts: Sequence[dict[str, Any] | str],
    ) -> dict[str, Any]:
        if self._last_strategy is None:
            raise RuntimeError("execute() must run before validate()")
        outputs: list[list[str]] = []
        for prompt in prompts:
            prompt_text = (
                prompt.get("prompt", "")
                if isinstance(prompt, dict)
                else str(prompt)
            )
            strategy = {
                **self._last_strategy,
                "prompt": prompt_text,
            }
            messages: list[str] = []
            async for event in self.execute(strategy):
                message = event.payload.get("message")
                if isinstance(message, str):
                    messages.append(message)
                if event.event_type is EventType.ERROR:
                    raise RuntimeError(event.payload["message"])
            outputs.append(messages)
        return {
            "backend": "gguf",
            "prompt_count": len(prompts),
            "outputs": outputs,
        }

    async def terminate(self) -> None:
        if self.process is None or self.process.returncode is not None:
            self._mark_terminated()
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3.0)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()
        self._mark_terminated()
