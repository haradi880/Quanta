"""Backend-neutral worker contract with no ML imports at module load."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from core.schemas import EventType, ProgressEvent, ValidationResult


class BaseWorker(ABC):
    """Uniform execute, validate, and terminate contract for every backend."""

    def __init__(self, job_id: UUID) -> None:
        self.job_id = job_id
        self._terminated = False

    def is_alive(self) -> bool:
        return not self._terminated

    def _mark_terminated(self) -> None:
        self._terminated = True

    @classmethod
    @abstractmethod
    def _import_backend(cls) -> Any:
        """Import and return backend symbols only when a worker is instantiated."""

    def _event(
        self,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> ProgressEvent:
        return ProgressEvent(
            job_id=self.job_id,
            event_type=event_type,
            timestamp_utc=datetime.now(timezone.utc),
            payload=payload,
            telemetry={},
        )

    def _error_event(self, backend: str, message: str) -> ProgressEvent:
        return self._event(
            EventType.ERROR,
            {
                "code": 500,
                "error": "backend_unavailable",
                "backend": backend,
                "message": message,
            },
        )

    @abstractmethod
    def execute(
        self,
        strategy_config: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        """Run conversion or inference and stream structured progress."""

    @abstractmethod
    async def validate(
        self,
        prompts: Sequence[dict[str, Any] | str],
    ) -> ValidationResult | dict[str, Any]:
        """Evaluate prompts with the worker's resulting artifact."""

    @abstractmethod
    async def terminate(self) -> None:
        """Stop all backend resources owned by this worker."""
