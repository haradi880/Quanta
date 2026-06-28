"""HaradiBots terminal interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from uuid import uuid4

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cli.doctor import run_offline_doctor
from core.auth_middleware import ensure_local_api_key
from core.orchestrator import process_job, purge as purge_runtime
from core.schemas import (
    AuthBlock,
    CallbackConfig,
    InterfaceType,
    JobEnvelope,
    JobMode,
    JobOperation,
    ModelSource,
    SystemPrompt,
    ValidationPolicy,
)
from telemetry.db import create_database, get_job

console = Console()


def build_envelope(args: argparse.Namespace) -> JobEnvelope:
    api_key = ensure_local_api_key()
    source = (
        ModelSource(local_path=args.model)
        if os.path.exists(args.model)
        else ModelSource(repo_id=args.model)
    )
    return JobEnvelope(
        schema_version="3.1",
        job_id=uuid4(),
        auth=AuthBlock(api_key=api_key),
        interface=InterfaceType.CLI,
        mode=JobMode(args.mode),
        operation=JobOperation.INFER,
        source_model=source,
        validation_policy=ValidationPolicy(),
        hardware_override=None,
        quantization_override=None,
        cluster_config=None,
        validation_prompts=None,
        system_prompt=SystemPrompt(preset_id=args.persona),
        telemetry_interval_ms=1000,
        callbacks=CallbackConfig(
            progress_channel="terminal",
            completion_channel="terminal",
        ),
    )


async def run_job(args: argparse.Namespace) -> int:
    envelope = build_envelope(args)
    exit_code = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Submitting job", total=None)
        async for event in process_job(envelope):
            state = event.payload.get("state", event.event_type.value)
            status = event.payload.get("status", "")
            progress.update(task, description=f"{state}: {status}".rstrip(": "))
            if event.event_type.value == "error":
                exit_code = 1
            if args.json:
                console.print_json(event.model_dump_json())
    return exit_code


def show_status(job_id: str) -> int:
    engine = create_database()
    try:
        row = get_job(engine, job_id)
    finally:
        engine.dispose()
    if row is None:
        console.print(f"Job {job_id} was not found.", style="red")
        return 1
    console.print(
        json.dumps(
            {
                "job_id": row.job_id,
                "model_source": row.model_source,
                "output_format": row.output_format,
                "state": row.state,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haradibots")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="submit a model job")
    run.add_argument("--model", required=True)
    run.add_argument("--mode", choices=("auto", "manual"), default="auto")
    run.add_argument("--persona", default="default")
    run.add_argument("--json", action="store_true")
    status = subparsers.add_parser("status", help="read local job metadata")
    status.add_argument("job_id")
    subparsers.add_parser("purge", help="invoke the purge controller")
    doctor = subparsers.add_parser(
        "doctor",
        help="verify the packaged offline native runtime",
    )
    doctor.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return asyncio.run(run_job(args))
    if args.command == "status":
        return show_status(args.job_id)
    if args.command == "doctor":
        try:
            result = asyncio.run(run_offline_doctor())
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
            if args.json:
                console.print_json(json.dumps(result))
            else:
                console.print(f"Offline runtime check failed: {exc}", style="red")
            return 1
        if args.json:
            console.print_json(json.dumps(result))
        else:
            console.print("Offline runtime check passed.", style="green")
        return 0
    console.print(
        "WARNING: purge terminates all workers and permanently deletes local "
        "models, outputs, validation data, credentials, logs, and job history.",
        style="bold red",
    )
    confirmation = console.input(
        "Type CONFIRM exactly to continue: "
    )
    if confirmation != "CONFIRM":
        console.print("Purge aborted; nothing was deleted.", style="yellow")
        return 2
    result = asyncio.run(purge_runtime())
    console.print_json(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
