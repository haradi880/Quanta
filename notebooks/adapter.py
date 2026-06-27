"""Kaggle and Colab notebook adapter for the Orchestrator event stream."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from uuid import uuid4

import nest_asyncio
from tqdm.auto import tqdm

from core.orchestrator import process_job
from core.schemas import (
    AuthBlock,
    CallbackConfig,
    InterfaceType,
    JobEnvelope,
    JobMode,
    ModelSource,
    ProgressEvent,
    SystemPrompt,
)


def notebook_environment() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "colab"
    return "jupyter"


def read_api_key(secret_name: str = "HARADIBOTS_API_KEY") -> str:
    if notebook_environment() == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(secret_name)
            if value:
                return value
        except (ImportError, KeyError):
            pass
    value = os.environ.get(secret_name)
    if not value:
        raise RuntimeError(f"notebook secret {secret_name} is required")
    return value


def build_envelope(model: str, mode: str = "auto") -> JobEnvelope:
    return JobEnvelope(
        schema_version="3.0",
        job_id=uuid4(),
        auth=AuthBlock(api_key=read_api_key()),
        interface=InterfaceType.KAGGLE,
        mode=JobMode(mode),
        model_source=ModelSource(repo_id=model),
        hardware_override=None,
        quantization_override=None,
        cluster_config=None,
        validation_prompts=None,
        system_prompt=SystemPrompt(preset_id="default"),
        telemetry_interval_ms=1000,
        callbacks=CallbackConfig(
            progress_channel="notebook",
            completion_channel="notebook",
        ),
    )


@dataclass
class NotebookResult:
    job_id: str
    events: list[ProgressEvent] = field(default_factory=list)

    @property
    def final_event(self) -> ProgressEvent | None:
        return self.events[-1] if self.events else None

    def _repr_html_(self) -> str:
        final_type = self.final_event.event_type.value if self.final_event else "none"
        return (
            "<div><strong>HaradiBots job:</strong> "
            f"{self.job_id}<br><strong>Events:</strong> {len(self.events)}"
            f"<br><strong>Final event:</strong> {final_type}</div>"
        )


async def run_job(model: str, mode: str = "auto") -> NotebookResult:
    envelope = build_envelope(model, mode)
    result = NotebookResult(job_id=str(envelope.job_id))
    progress = tqdm(desc="HaradiBots", unit="event")
    try:
        async for event in process_job(envelope):
            result.events.append(event)
            progress.set_description(
                str(event.payload.get("state", event.event_type.value))
            )
            progress.update(1)
    finally:
        progress.close()
    return result


def run(model: str, mode: str = "auto") -> NotebookResult:
    """Run from a notebook cell, including one with an active event loop."""

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_job(model, mode))
