"""AutoAWQ quantization worker with guarded, lazy backend loading."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from core.schemas import EventType, ProgressEvent
from engines.base_worker import BaseWorker


class AWQWorker(BaseWorker):
    def __init__(self, job_id: UUID) -> None:
        super().__init__(job_id)
        self._backend: tuple[Any, Any] | None = None
        self._backend_error: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        try:
            self._backend = self._import_backend()
        except Exception as exc:
            self._backend_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def _import_backend(cls) -> tuple[Any, Any]:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        return AutoAWQForCausalLM, AutoTokenizer

    @staticmethod
    def _quantize(
        model_class: Any,
        tokenizer_class: Any,
        strategy_config: dict[str, Any],
    ) -> tuple[Any, Any, str]:
        model_source = strategy_config.get(
            "model_source",
            strategy_config.get("model_path"),
        )
        output_path = strategy_config.get("output_path")
        if not isinstance(model_source, str) or not model_source:
            raise ValueError("AWQ strategy requires model_source or model_path")
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("AWQ strategy requires output_path")

        load_kwargs = strategy_config.get("load_kwargs", {})
        if not isinstance(load_kwargs, dict):
            raise ValueError("load_kwargs must be an object")
        quant_config = strategy_config.get(
            "quant_config",
            {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            },
        )
        if not isinstance(quant_config, dict):
            raise ValueError("quant_config must be an object")

        model = model_class.from_pretrained(model_source, **load_kwargs)
        tokenizer = tokenizer_class.from_pretrained(model_source)
        model.quantize(tokenizer, quant_config=quant_config)
        destination = Path(output_path)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_quantized(str(destination))
        tokenizer.save_pretrained(str(destination))
        return model, tokenizer, str(destination.resolve())

    async def execute(
        self,
        strategy_config: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        if self._backend is None:
            yield self._error_event(
                "awq",
                self._backend_error or "AutoAWQ backend is unavailable",
            )
            return

        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {"backend": "awq", "status": "started", "progress_pct": 0.0},
        )
        model_class, tokenizer_class = self._backend
        try:
            self._model, self._tokenizer, output_path = await asyncio.to_thread(
                self._quantize,
                model_class,
                tokenizer_class,
                dict(strategy_config),
            )
        except Exception as exc:
            yield self._error_event(
                "awq",
                f"{type(exc).__name__}: {exc}",
            )
            return
        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {
                "backend": "awq",
                "status": "complete",
                "progress_pct": 100.0,
                "output_path": output_path,
            },
        )

    async def validate(
        self,
        prompts: Sequence[dict[str, Any] | str],
    ) -> dict[str, Any]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("AWQ model is not loaded")

        def infer() -> list[str]:
            outputs: list[str] = []
            for prompt in prompts:
                text = (
                    prompt.get("prompt", "")
                    if isinstance(prompt, dict)
                    else str(prompt)
                )
                encoded = self._tokenizer(text, return_tensors="pt")
                generated = self._model.generate(**encoded)
                outputs.append(
                    self._tokenizer.decode(
                        generated[0],
                        skip_special_tokens=True,
                    )
                )
            return outputs

        return {
            "backend": "awq",
            "outputs": await asyncio.to_thread(infer),
        }

    async def terminate(self) -> None:
        self._model = None
        self._tokenizer = None
        self._mark_terminated()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return
