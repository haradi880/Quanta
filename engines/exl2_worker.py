"""EXL2 conversion and inference worker with lazy backend imports."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from core.schemas import EventType, ProgressEvent
from engines.base_worker import BaseWorker


class EXL2Worker(BaseWorker):
    def __init__(self, job_id: UUID) -> None:
        super().__init__(job_id)
        self._backend: tuple[Any, ...] | None = None
        self._backend_error: str | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._output_path: str | None = None
        self._runtime: tuple[Any, Any] | None = None
        try:
            self._backend = self._import_backend()
        except Exception as exc:
            self._backend_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def _import_backend(cls) -> tuple[Any, ...]:
        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2BaseGenerator

        return (
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
            ExLlamaV2BaseGenerator,
        )

    @staticmethod
    def _conversion_command(strategy_config: dict[str, Any]) -> list[str]:
        script_value = strategy_config.get("convert_script") or os.environ.get(
            "HARADIBOTS_EXL2_CONVERT_SCRIPT"
        )
        if not isinstance(script_value, str) or not Path(script_value).is_file():
            raise FileNotFoundError(
                "EXL2 convert.py is not configured; set "
                "HARADIBOTS_EXL2_CONVERT_SCRIPT"
            )
        input_path = strategy_config.get(
            "model_path",
            strategy_config.get("model_source"),
        )
        work_path = strategy_config.get("work_path")
        output_path = strategy_config.get("output_path")
        if not all(
            isinstance(value, str) and value
            for value in (input_path, work_path, output_path)
        ):
            raise ValueError(
                "EXL2 strategy requires model_path, work_path, and output_path"
            )
        bits = float(strategy_config.get("bits", 4.0))
        if not 2.0 <= bits <= 8.0:
            raise ValueError("EXL2 target bits must be between 2 and 8")

        command = [
            sys.executable,
            str(Path(script_value).resolve()),
            "-i",
            str(Path(input_path).resolve()),
            "-o",
            str(Path(work_path).resolve()),
            "-cf",
            str(Path(output_path).resolve()),
            "-b",
            str(bits),
        ]
        if strategy_config.get("no_resume", False):
            command.append("-nr")
        calibration = strategy_config.get("calibration_dataset")
        if isinstance(calibration, str):
            command.extend(["-c", str(Path(calibration).resolve())])
        return command

    async def execute(
        self,
        strategy_config: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        if self._backend is None:
            yield self._error_event(
                "exl2",
                self._backend_error or "ExLlamaV2 backend is unavailable",
            )
            return
        try:
            command = self._conversion_command(strategy_config)
            self._output_path = str(Path(strategy_config["output_path"]).resolve())
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, ValueError) as exc:
            yield self._error_event("exl2", str(exc))
            return

        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {"backend": "exl2", "status": "launched", "pid": self.process.pid},
        )
        if self.process.stdout is None:
            yield self._error_event("exl2", "converter stdout pipe was not created")
            return
        async for raw_line in self.process.stdout:
            yield self._event(
                EventType.QUANTIZATION_PROGRESS,
                {
                    "backend": "exl2",
                    "status": "running",
                    "message": raw_line.decode(
                        "utf-8",
                        errors="replace",
                    ).rstrip(),
                },
            )
        return_code = await self.process.wait()
        if return_code:
            yield self._error_event(
                "exl2",
                f"conversion process exited with code {return_code}",
            )
            return
        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {
                "backend": "exl2",
                "status": "complete",
                "progress_pct": 100.0,
                "output_path": self._output_path,
            },
        )

    def _load_runtime(self) -> tuple[Any, Any]:
        if self._backend is None or self._output_path is None:
            raise RuntimeError("EXL2 model has not been converted")
        (
            model_class,
            cache_class,
            config_class,
            tokenizer_class,
            generator_class,
        ) = self._backend
        config = config_class()
        config.model_dir = self._output_path
        config.prepare()
        model = model_class(config)
        cache = cache_class(model, lazy=True)
        model.load_autosplit(cache)
        tokenizer = tokenizer_class(config)
        generator = generator_class(model, cache, tokenizer)
        generator.warmup()
        self._runtime = (model, generator)
        return self._runtime

    async def validate(
        self,
        prompts: Sequence[dict[str, Any] | str],
    ) -> dict[str, Any]:
        if self._runtime is None:
            await asyncio.to_thread(self._load_runtime)
        if self._runtime is None:
            raise RuntimeError("EXL2 runtime failed to initialize")
        generator = self._runtime[1]

        def infer() -> list[str]:
            return [
                generator.generate_simple(
                    (
                        prompt.get("prompt", "")
                        if isinstance(prompt, dict)
                        else str(prompt)
                    ),
                    max_new_tokens=128,
                )
                for prompt in prompts
            ]

        return {
            "backend": "exl2",
            "outputs": await asyncio.to_thread(infer),
        }

    async def terminate(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self._runtime = None
        self._mark_terminated()
