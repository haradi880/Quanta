"""GGUF execution through an isolated llama.cpp subprocess."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
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

    def _quantization_commands(
        self,
        strategy_config: dict[str, Any],
    ) -> tuple[list[str] | None, list[str], str]:
        converter = strategy_config.get("convert_script") or os.environ.get(
            "HARADIBOTS_GGUF_CONVERT_SCRIPT"
        )
        quantizer = strategy_config.get("quantize_binary") or os.environ.get(
            "HARADIBOTS_LLAMA_QUANTIZE_BIN"
        )
        source = strategy_config.get("model_path")
        output = strategy_config.get("output_path")
        work = strategy_config.get("work_path")
        target_format = strategy_config.get("format")
        if not quantizer or not Path(quantizer).is_file():
            raise FileNotFoundError(
                "GGUF quantization requires HARADIBOTS_LLAMA_QUANTIZE_BIN"
            )
        if not source or not Path(source).exists():
            raise FileNotFoundError("GGUF conversion source does not exist")
        if not output or not work or not target_format:
            raise ValueError("GGUF conversion requires output_path, work_path, and format")
        work_path = Path(work)
        work_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = Path(source)
        if source_path.is_file() and source_path.suffix.lower() == ".gguf":
            convert_command = None
            intermediate = source_path
        else:
            if not source_path.is_dir():
                raise FileNotFoundError(
                    "GGUF conversion source must be a model directory or GGUF file"
                )
            if not converter or not Path(converter).is_file():
                raise FileNotFoundError(
                    "GGUF conversion requires HARADIBOTS_GGUF_CONVERT_SCRIPT"
                )
            intermediate = work_path / "model-f16.gguf"
            convert_command = [
                sys.executable,
                str(Path(converter).resolve()),
                str(source_path.resolve()),
                "--outfile",
                str(intermediate.resolve()),
                "--outtype",
                "f16",
            ]
        quantize_command = [
            str(Path(quantizer).resolve()),
            str(intermediate.resolve()),
            str(output_path.resolve()),
            str(target_format),
        ]
        return convert_command, quantize_command, str(output_path.resolve())

    async def _run_stage(
        self,
        command: list[str],
        stage: str,
    ) -> AsyncIterator[ProgressEvent]:
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if self.process.stdout is None:
            raise RuntimeError(f"{stage} stdout pipe was not created")
        async for raw_line in self.process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            payload: dict[str, Any] = {
                "backend": "gguf",
                "stage": stage,
                "status": "running",
                "message": line,
            }
            match = PERCENT_PATTERN.search(line)
            if match:
                payload["progress_pct"] = min(max(float(match.group(1)), 0.0), 100.0)
            yield self._event(EventType.QUANTIZATION_PROGRESS, payload)
        return_code = await self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"{stage} subprocess exited with code {return_code}")

    async def execute(
        self,
        strategy_config: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        self._last_strategy = dict(strategy_config)
        if strategy_config.get("operation") == "quantize":
            try:
                convert, quantize, output_path = self._quantization_commands(
                    strategy_config
                )
                if convert is not None:
                    async for event in self._run_stage(convert, "convert_f16"):
                        yield event
                async for event in self._run_stage(quantize, "quantize"):
                    yield event
                if not Path(output_path).is_file() or Path(output_path).stat().st_size <= 0:
                    raise RuntimeError("GGUF quantizer produced no artifact")
                yield self._event(
                    EventType.QUANTIZATION_PROGRESS,
                    {
                        "backend": "gguf",
                        "status": "complete",
                        "progress_pct": 100.0,
                        "output_path": output_path,
                    },
                )
            except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
                yield self._error_event("gguf", str(exc))
            return
        progress_event = (
            EventType.INFERENCE_PROGRESS
            if strategy_config.get("operation") == "infer"
            else EventType.QUANTIZATION_PROGRESS
        )
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
            progress_event,
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
            yield self._event(progress_event, payload)

        return_code = await self.process.wait()
        if return_code != 0:
            yield self._error_event(
                "gguf",
                f"llama.cpp subprocess exited with code {return_code}",
            )
            return
        yield self._event(
            progress_event,
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
