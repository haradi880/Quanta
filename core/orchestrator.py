"""Authenticated job orchestration and mandatory process teardown."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

import psutil

from core.accelerator import select_strategy
from core.artifacts import acquire_gguf_artifact, inspect_gguf_metadata
from core.auth_middleware import authenticate
from core.hf_inspector import inspect_repo
from core.profiler import snapshot
from core.schemas import (
    ErrorEnvelope,
    EventType,
    HardwareProfile,
    JobEnvelope,
    ProgressEvent,
    SCHEMA_VERSION,
    TeardownComplete,
    ValidationResult,
)


LOGGER = logging.getLogger(__name__)


class JobState(StrEnum):
    IDLE = "IDLE"
    PROFILING = "PROFILING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    TEARDOWN = "TEARDOWN"
    ERROR = "ERROR"
    CLUSTER_DISPATCH = "CLUSTER_DISPATCH"


class StateMachineError(RuntimeError):
    """Raised when an event is invalid for the current job state."""


TRANSITION_TABLE: dict[tuple[JobState, str], JobState] = {
    (JobState.IDLE, "job_received"): JobState.PROFILING,
    (JobState.PROFILING, "profile_complete"): JobState.PLANNING,
    (JobState.PROFILING, "failed"): JobState.ERROR,
    (JobState.PLANNING, "plan_complete"): JobState.EXECUTING,
    (JobState.PLANNING, "cluster_required"): JobState.CLUSTER_DISPATCH,
    (JobState.PLANNING, "failed"): JobState.ERROR,
    (JobState.CLUSTER_DISPATCH, "dispatch_complete"): JobState.EXECUTING,
    (JobState.CLUSTER_DISPATCH, "failed"): JobState.ERROR,
    (JobState.EXECUTING, "execution_complete"): JobState.VALIDATING,
    (JobState.EXECUTING, "failed"): JobState.ERROR,
    (JobState.VALIDATING, "validation_complete"): JobState.TEARDOWN,
    (JobState.VALIDATING, "failed"): JobState.ERROR,
    (JobState.ERROR, "begin_teardown"): JobState.TEARDOWN,
    (JobState.TEARDOWN, "teardown_complete"): JobState.IDLE,
}


def transition(current_state: JobState, event: str) -> JobState:
    """Return the next state or reject an undefined transition."""

    try:
        return TRANSITION_TABLE[(current_state, event)]
    except KeyError as exc:
        raise StateMachineError(
            f"invalid transition from {current_state.value} on event '{event}'"
        ) from exc


class Orchestrator:
    """Own job state, worker handles, and the mandatory teardown boundary."""

    def __init__(self, *, teardown_grace_seconds: float = 3.0) -> None:
        if teardown_grace_seconds < 0:
            raise ValueError("teardown_grace_seconds must be non-negative")
        self._teardown_grace_seconds = teardown_grace_seconds
        self._worker_registry: dict[UUID, list[Any]] = {}
        self._job_states: dict[UUID, JobState] = {}
        self._state_history: dict[UUID, list[JobState]] = {}

    def _set_initial_state(self, job_id: UUID) -> None:
        self._job_states[job_id] = JobState.IDLE
        self._state_history[job_id] = [JobState.IDLE]

    def _transition_job(self, job_id: UUID, event: str) -> JobState:
        current = self._job_states[job_id]
        next_state = transition(current, event)
        self._job_states[job_id] = next_state
        self._state_history[job_id].append(next_state)
        return next_state

    def state_history(self, job_id: UUID) -> tuple[JobState, ...]:
        return tuple(self._state_history.get(job_id, ()))

    def register_worker(self, job_id: UUID, handle: Any) -> None:
        """Register every process or actor handle before execution begins."""

        handles = self._worker_registry.setdefault(job_id, [])
        if any(existing is handle for existing in handles):
            raise ValueError(f"worker handle is already registered for job {job_id}")
        handles.append(handle)

    def worker_handles(self, job_id: UUID) -> tuple[Any, ...]:
        """Return an immutable view of handles currently owned by a job."""

        return tuple(self._worker_registry.get(job_id, ()))

    def _create_and_register_worker(
        self,
        envelope: JobEnvelope,
        strategy: dict[str, Any],
    ) -> Any:
        """Instantiate only the selected backend and register it immediately."""

        backend = str(strategy["backend"]).lower()
        target_format = str(strategy["format"]).lower()
        if "llama.cpp" in backend:
            from engines.gguf_worker import GGUFWorker

            worker: Any = GGUFWorker(envelope.job_id)
        elif "autoawq" in backend:
            from engines.awq_worker import AWQWorker

            worker = AWQWorker(envelope.job_id)
        elif backend == "vllm/exl2" and "awq" not in target_format:
            from engines.exl2_worker import EXL2Worker

            worker = EXL2Worker(envelope.job_id)
        elif "vllm" in backend:
            from engines.vllm_worker import VLLMWorker

            worker = VLLMWorker(envelope.job_id)
        elif "exl2" in backend:
            from engines.exl2_worker import EXL2Worker

            worker = EXL2Worker(envelope.job_id)
        else:
            raise RuntimeError(f"no worker class is mapped for backend '{backend}'")
        self.register_worker(envelope.job_id, worker)
        return worker

    @staticmethod
    def _execution_strategy(
        envelope: JobEnvelope,
        strategy: dict[str, Any],
    ) -> dict[str, Any]:
        execution_strategy = dict(strategy)
        if envelope.model_source.repo_id is not None:
            execution_strategy["model_source"] = envelope.model_source.repo_id
        if envelope.model_source.local_path is not None:
            execution_strategy["model_path"] = envelope.model_source.local_path

        cache_root = Path(
            os.environ.get(
                "HARADIBOTS_CACHE_ROOT",
                str(Path.home() / ".haradibots" / "cache"),
            )
        ).expanduser()
        job_root = cache_root / str(envelope.job_id)
        execution_strategy.setdefault("work_path", str(job_root / "work"))
        execution_strategy.setdefault("output_path", str(job_root / "output"))
        return execution_strategy

    @staticmethod
    def _event(
        envelope: JobEnvelope,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> ProgressEvent:
        return ProgressEvent(
            schema_version=SCHEMA_VERSION,
            job_id=envelope.job_id,
            event_type=event_type,
            timestamp_utc=datetime.now(timezone.utc),
            payload=payload,
            telemetry={},
        )

    @staticmethod
    def _process_for(handle: Any) -> Any:
        process = getattr(handle, "process", None)
        return process if process is not None else handle

    @classmethod
    def _pid_for(cls, handle: Any) -> int | None:
        process = cls._process_for(handle)
        pid = getattr(process, "pid", None)
        return int(pid) if isinstance(pid, int) else None

    @classmethod
    def _is_alive(cls, handle: Any) -> bool:
        process = cls._process_for(handle)
        is_alive = getattr(process, "is_alive", None)
        if callable(is_alive):
            return bool(is_alive())
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
        if hasattr(process, "returncode"):
            return getattr(process, "returncode") is None
        return True

    @classmethod
    async def _call_process_method(cls, handle: Any, method_name: str) -> None:
        process = cls._process_for(handle)
        method = getattr(process, method_name, None)
        if method is None:
            method = getattr(handle, method_name, None)
        if not callable(method):
            return
        if inspect.iscoroutinefunction(method):
            await method()
            return
        result = method()
        if inspect.isawaitable(result):
            await result

    @classmethod
    async def _wait_for_exit(cls, handle: Any) -> None:
        process = cls._process_for(handle)
        wait = getattr(process, "wait", None)
        if not callable(wait):
            while cls._is_alive(handle):
                await asyncio.sleep(0.01)
            return
        if inspect.iscoroutinefunction(wait):
            await wait()
            return
        await asyncio.to_thread(wait)

    def _enumerate_job_handles(self, job_id: UUID) -> list[Any]:
        """Return registered roots plus every discoverable subprocess descendant."""

        roots = list(self._worker_registry.get(job_id, []))
        handles = list(roots)
        known_pids = {
            pid
            for pid in (self._pid_for(handle) for handle in roots)
            if pid is not None
        }

        def append_children(process: psutil.Process) -> None:
            try:
                children = process.children(recursive=False)
            except (psutil.Error, OSError):
                return
            for child in children:
                if child.pid in known_pids:
                    continue
                known_pids.add(child.pid)
                handles.append(child)
                append_children(child)

        for root in roots:
            pid = self._pid_for(root)
            if pid is None:
                continue
            try:
                append_children(psutil.Process(pid))
            except (psutil.Error, OSError):
                continue
        return handles

    async def _teardown(self, job_id: UUID) -> ProgressEvent:
        """Harvest every registered worker using TERM then KILL escalation."""

        handles = self._enumerate_job_handles(job_id)
        harvested_pids = [
            pid
            for pid in (self._pid_for(handle) for handle in handles)
            if pid is not None
        ]

        for handle in reversed(handles):
            if self._is_alive(handle):
                await self._call_process_method(handle, "terminate")

        if handles:
            waits = [
                asyncio.create_task(self._wait_for_exit(handle))
                for handle in handles
                if self._is_alive(handle)
            ]
            if waits:
                done, pending = await asyncio.wait(
                    waits,
                    timeout=self._teardown_grace_seconds,
                )
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                completed_results = await asyncio.gather(
                    *done,
                    return_exceptions=True,
                )
                for result in completed_results:
                    if isinstance(result, BaseException):
                        LOGGER.warning(
                            "worker wait raised during teardown: %s",
                            result,
                        )

        survivors = [handle for handle in handles if self._is_alive(handle)]
        for handle in reversed(survivors):
            await self._call_process_method(handle, "kill")

        forced_kill_count = len(survivors)
        if forced_kill_count:
            LOGGER.warning(
                "job %s required %d forced worker kill(s)",
                job_id,
                forced_kill_count,
            )

        self._worker_registry.pop(job_id, None)
        manifest = TeardownComplete(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            harvested_pids=harvested_pids,
            forced_kill_count=forced_kill_count,
            timestamp_utc=datetime.now(timezone.utc),
        )
        return ProgressEvent(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            event_type=EventType.TEARDOWN_COMPLETE,
            timestamp_utc=manifest.timestamp_utc,
            payload=manifest.model_dump(mode="json"),
            telemetry={},
        )

    async def _execute_registered_workers(
        self,
        envelope: JobEnvelope,
        strategy: dict[str, Any],
    ) -> AsyncIterator[ProgressEvent]:
        workers = self._worker_registry.get(envelope.job_id, [])
        if not workers:
            raise RuntimeError(
                "no execution worker is registered; execution backends arrive in Phase 6"
            )

        for worker in workers:
            execute = getattr(worker, "execute", None)
            if not callable(execute):
                raise RuntimeError("registered worker has no execute() method")
            stream = execute(strategy)
            if inspect.isawaitable(stream):
                stream = await stream
            if not hasattr(stream, "__aiter__"):
                raise RuntimeError("worker execute() must return an async iterator")
            async for event in stream:
                if not isinstance(event, ProgressEvent):
                    raise RuntimeError("worker emitted a non-ProgressEvent value")
                yield event
                if event.event_type is EventType.ERROR:
                    raise RuntimeError(
                        str(event.payload.get("message", "worker execution failed"))
                    )

    async def _validate_registered_workers(
        self,
        envelope: JobEnvelope,
    ) -> dict[str, Any]:
        workers = self._worker_registry.get(envelope.job_id, [])
        if not workers:
            raise RuntimeError("no worker is available for validation")
        validate = getattr(workers[0], "validate", None)
        if not callable(validate):
            raise RuntimeError("registered worker has no validate() method")
        prompts = [
            prompt.model_dump(mode="json")
            for prompt in (envelope.validation_prompts or [])
        ]
        result = validate(prompts)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="python")
        if not isinstance(result, dict):
            raise RuntimeError("worker validate() returned an unsupported value")
        try:
            validated = ValidationResult.model_validate(result)
        except Exception as exc:
            raise RuntimeError(
                "backend did not produce the required original-vs-quantized "
                "ValidationResult; artifact delivery is blocked"
            ) from exc
        return validated.model_dump(mode="json")

    async def process_job(
        self,
        envelope: JobEnvelope,
    ) -> AsyncIterator[ProgressEvent]:
        """Authenticate, route the FSM, and never skip teardown after entry."""

        authentication = authenticate(envelope)
        if isinstance(authentication, ErrorEnvelope):
            yield self._event(
                envelope,
                EventType.ERROR,
                authentication.model_dump(mode="json"),
            )
            return

        self._set_initial_state(envelope.job_id)
        entered_state_machine = False
        succeeded = False
        stream_closing = False
        try:
            state = self._transition_job(envelope.job_id, "job_received")
            entered_state_machine = True
            yield self._event(
                envelope,
                EventType.HARDWARE_PROFILE,
                {"state": state.value, "status": "started"},
            )

            hardware = snapshot()
            hardware_payload = HardwareProfile.model_validate(hardware).model_dump(
                mode="json"
            )
            state = self._transition_job(envelope.job_id, "profile_complete")
            yield self._event(
                envelope,
                EventType.HARDWARE_PROFILE,
                {
                    "state": state.value,
                    "status": "complete",
                    "hardware_profile": hardware_payload,
                },
            )

            if envelope.model_source.local_path is not None:
                local_path = Path(envelope.model_source.local_path)
                if not local_path.exists():
                    raise FileNotFoundError(
                        f"local model path does not exist: {local_path}"
                    )
                raise RuntimeError(
                    "local model inspection is not implemented in the "
                    "Hugging Face metadata phase"
                )

            repo_id = envelope.model_source.repo_id
            if repo_id is None:
                raise RuntimeError("model source contains no repository identifier")
            model_meta = await inspect_repo(repo_id)
            if not model_meta["repo_exists"]:
                raise FileNotFoundError(
                    f"Hugging Face repository does not exist: "
                    f"{repo_id}"
                )
            if model_meta["parameter_count"] is None:
                raise RuntimeError(
                    "model parameter count is unavailable; strategy cannot be planned"
                )

            acquired_model_path = None
            if any(
                filename.lower().endswith(".gguf")
                for filename in model_meta["file_manifest"]
            ):
                acquired_model_path = await acquire_gguf_artifact(
                    repo_id,
                    model_meta,
                    str(model_meta.get("quant_format") or "GGUF"),
                    revision=envelope.model_source.revision,
                )
                gguf_metadata = inspect_gguf_metadata(acquired_model_path)
                for field_name, value in gguf_metadata.items():
                    if value is not None:
                        model_meta[field_name] = value

            strategy = select_strategy(
                hardware,
                model_meta,
                mode=envelope.mode.value,
                override=(
                    envelope.quantization_override.model_dump(mode="python")
                    if envelope.quantization_override is not None
                    else None
                ),
            )
            if acquired_model_path is not None:
                strategy["format"] = str(
                    model_meta.get("quant_format") or strategy["format"]
                )
                strategy["backend"] = (
                    "llama.cpp CUDA"
                    if int(hardware.get("gpu_count", 0)) > 0
                    else "llama.cpp"
                )
            elif "llama.cpp" in str(strategy["backend"]).lower():
                acquired_model_path = await acquire_gguf_artifact(
                    repo_id,
                    model_meta,
                    str(strategy["format"]),
                    revision=envelope.model_source.revision,
                )
            if envelope.cluster_config is not None:
                state = self._transition_job(
                    envelope.job_id,
                    "cluster_required",
                )
                yield self._event(
                    envelope,
                    EventType.CLUSTER_NODE_STATUS,
                    {"state": state.value, "status": "dispatching"},
                )
                raise RuntimeError("cluster dispatch backends arrive in Phase 11")

            state = self._transition_job(envelope.job_id, "plan_complete")
            execution_strategy = self._execution_strategy(envelope, strategy)
            if acquired_model_path is not None:
                execution_strategy["model_path"] = acquired_model_path
                execution_strategy.setdefault(
                    "prompt",
                    "Reply with the single word OK.",
                )
                execution_strategy.setdefault("max_tokens", 16)
            if not self.worker_handles(envelope.job_id):
                self._create_and_register_worker(
                    envelope,
                    execution_strategy,
                )
            yield self._event(
                envelope,
                EventType.STRATEGY_SELECTED,
                {
                    "state": state.value,
                    "status": "complete",
                    "strategy_config": execution_strategy,
                },
            )

            async for worker_event in self._execute_registered_workers(
                envelope,
                execution_strategy,
            ):
                yield worker_event
            state = self._transition_job(envelope.job_id, "execution_complete")
            yield self._event(
                envelope,
                EventType.QUANTIZATION_PROGRESS,
                {"state": state.value, "status": "execution_complete"},
            )

            validation = await self._validate_registered_workers(envelope)
            yield self._event(
                envelope,
                EventType.VALIDATION_RESULT,
                {
                    "state": state.value,
                    "status": "complete",
                    "validation_result": validation,
                },
            )
            self._transition_job(envelope.job_id, "validation_complete")
            succeeded = True
        except (GeneratorExit, asyncio.CancelledError):
            stream_closing = True
            current = self._job_states[envelope.job_id]
            if current is not JobState.ERROR:
                self._transition_job(envelope.job_id, "failed")
            raise
        except Exception as exc:
            current = self._job_states[envelope.job_id]
            if current is not JobState.ERROR:
                self._transition_job(envelope.job_id, "failed")
            yield self._event(
                envelope,
                EventType.ERROR,
                {
                    "state": JobState.ERROR.value,
                    "code": 500,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
        finally:
            if entered_state_machine:
                current = self._job_states[envelope.job_id]
                if current is JobState.ERROR:
                    self._transition_job(envelope.job_id, "begin_teardown")
                elif current is not JobState.TEARDOWN:
                    # A cancellation or unexpected path still enters TEARDOWN.
                    self._job_states[envelope.job_id] = JobState.TEARDOWN
                    self._state_history[envelope.job_id].append(JobState.TEARDOWN)
                teardown_event = await self._teardown(envelope.job_id)
                state = self._transition_job(
                    envelope.job_id,
                    "teardown_complete",
                )
                if not stream_closing:
                    yield teardown_event
                    yield self._event(
                        envelope,
                        EventType.COMPLETE,
                        {
                            "state": state.value,
                            "status": "complete" if succeeded else "failed",
                        },
                    )


_DEFAULT_ORCHESTRATOR = Orchestrator()


async def process_job(envelope: JobEnvelope) -> AsyncIterator[ProgressEvent]:
    """Module-level entry point used by every interface shell."""

    async for event in _DEFAULT_ORCHESTRATOR.process_job(envelope):
        yield event
