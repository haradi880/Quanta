"""vLLM tensor-parallel worker and native tokenizer client."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import UUID

import aiohttp

from core.schemas import EventType, ProgressEvent
from engines.base_worker import BaseWorker


class VLLMWorker(BaseWorker):
    def __init__(self, job_id: UUID) -> None:
        super().__init__(job_id)
        self._backend: tuple[Any, Any] | None = None
        self._backend_error: str | None = None
        self._llm: Any = None
        self._sampling_params_class: Any = None
        self._backend_url: str | None = None
        self._model_name: str | None = None
        try:
            self._backend = self._import_backend()
        except Exception as exc:
            self._backend_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def _import_backend(cls) -> tuple[Any, Any]:
        from vllm import LLM, SamplingParams

        return LLM, SamplingParams

    async def count_tokens(
        self,
        text: str,
        backend_url: str,
        *,
        model: str | None = None,
    ) -> int:
        """Return the exact count from the running vLLM `/tokenize` endpoint."""

        payload: dict[str, Any] = {"prompt": text}
        if model:
            payload["model"] = model
        url = f"{backend_url.rstrip('/')}/tokenize"
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        detail = (await response.text())[:200]
                        raise RuntimeError(
                            f"vLLM /tokenize returned HTTP {response.status}: {detail}"
                        )
                    data = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError("vLLM /tokenize request failed") from exc

        if isinstance(data, dict):
            count = data.get("count")
            if isinstance(count, int) and count >= 0:
                return count
            for key in ("token_ids", "tokens"):
                values = data.get(key)
                if isinstance(values, list):
                    return len(values)
        raise RuntimeError("vLLM /tokenize response contains no token count")

    @staticmethod
    def _load_llm(
        llm_class: Any,
        model_source: str,
        tensor_parallel_size: int,
        kwargs: dict[str, Any],
    ) -> Any:
        return llm_class(
            model=model_source,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )

    async def execute(
        self,
        strategy_config: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        if self._backend is None:
            yield self._error_event(
                "vllm",
                self._backend_error or "vLLM backend is unavailable",
            )
            return

        model_source = strategy_config.get(
            "model_source",
            strategy_config.get("model_path"),
        )
        if not isinstance(model_source, str) or not model_source:
            yield self._error_event(
                "vllm",
                "vLLM strategy requires model_source or model_path",
            )
            return
        tensor_parallel_size = int(strategy_config.get("tp_degree", 1))
        if tensor_parallel_size < 1:
            yield self._error_event(
                "vllm",
                "tensor parallel degree must be positive",
            )
            return
        engine_kwargs = strategy_config.get("engine_kwargs", {})
        if not isinstance(engine_kwargs, dict):
            yield self._error_event("vllm", "engine_kwargs must be an object")
            return

        self._backend_url = strategy_config.get("backend_url")
        self._model_name = model_source
        llm_class, self._sampling_params_class = self._backend
        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {
                "backend": "vllm",
                "status": "loading",
                "tensor_parallel_size": tensor_parallel_size,
            },
        )
        try:
            self._llm = await asyncio.to_thread(
                self._load_llm,
                llm_class,
                model_source,
                tensor_parallel_size,
                engine_kwargs,
            )
        except Exception as exc:
            yield self._error_event("vllm", f"{type(exc).__name__}: {exc}")
            return
        yield self._event(
            EventType.QUANTIZATION_PROGRESS,
            {
                "backend": "vllm",
                "status": "complete",
                "progress_pct": 100.0,
                "tensor_parallel_size": tensor_parallel_size,
            },
        )

    async def validate(
        self,
        prompts: Sequence[dict[str, Any] | str],
    ) -> dict[str, Any]:
        if self._llm is None or self._sampling_params_class is None:
            raise RuntimeError("vLLM engine is not loaded")
        prompt_texts = [
            prompt.get("prompt", "") if isinstance(prompt, dict) else str(prompt)
            for prompt in prompts
        ]
        token_counts = None
        if self._backend_url:
            token_counts = [
                await self.count_tokens(
                    text,
                    self._backend_url,
                    model=self._model_name,
                )
                for text in prompt_texts
            ]
        sampling = self._sampling_params_class(max_tokens=128)
        results = await asyncio.to_thread(
            self._llm.generate,
            prompt_texts,
            sampling,
        )
        outputs = [
            result.outputs[0].text if getattr(result, "outputs", None) else ""
            for result in results
        ]
        return {
            "backend": "vllm",
            "outputs": outputs,
            "native_token_counts": token_counts,
        }

    async def terminate(self) -> None:
        self._llm = None
        self._mark_terminated()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return
