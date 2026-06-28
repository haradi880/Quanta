import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from cli.main import build_envelope as build_cli_envelope
from cli.main import build_parser
from core.schemas import EventType, ProgressEvent
from notebooks.adapter import build_envelope as build_notebook_envelope
from ui.app import build_envelope as build_gui_envelope


def event_for(job_id):
    return ProgressEvent(
        schema_version="3.0",
        job_id=job_id,
        event_type=EventType.COMPLETE,
        timestamp_utc=datetime.now(timezone.utc),
        payload={"state": "IDLE", "status": "complete"},
        telemetry={},
    )


def test_cli_builds_exact_authenticated_envelope(monkeypatch):
    monkeypatch.setattr("cli.main.ensure_local_api_key", lambda: "test-key")
    args = build_parser().parse_args(["run", "--model", "owner/model"])

    envelope = build_cli_envelope(args)

    assert envelope.auth.api_key == "test-key"
    assert envelope.interface.value == "cli"


def test_api_rejects_missing_header():
    from cluster.api_server import app

    response = TestClient(app).post("/jobs", json={})

    assert response.status_code == 401
    assert response.json()["error"] == "authentication_failed"


def test_api_returns_sse_for_valid_envelope(monkeypatch):
    import cluster.api_server as server

    monkeypatch.setattr(
        server,
        "validate_api_key",
        lambda value: {"subject": "test", "auth_type": "api_key"},
    )
    envelope = build_gui_envelope("owner/model", "auto", "placeholder.jwt")
    payload = envelope.model_dump(mode="json")
    payload["auth"] = {"api_key": "test-key", "jwt_token": None}
    payload["interface"] = "api"

    async def fake_process_job(job):
        yield event_for(job.job_id)

    monkeypatch.setattr(server, "process_job", fake_process_job)
    response = TestClient(server.app).post(
        "/jobs",
        headers={"X-API-Key": "test-key"},
        json=payload,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"event_type":"complete"' in response.text


def test_gui_and_notebook_use_their_contract_interfaces(monkeypatch):
    gui = build_gui_envelope("owner/model", "auto", "placeholder.jwt")
    monkeypatch.setattr(
        "notebooks.adapter.ensure_local_api_key",
        lambda: "test-key",
    )
    notebook = build_notebook_envelope("owner/model")

    assert gui.interface.value == "gui"
    assert gui.auth.jwt_token == "placeholder.jwt"
    assert notebook.interface.value == "kaggle"
    assert notebook.auth.api_key == "test-key"


def test_interface_modules_never_import_engines():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "cli/main.py",
        "cluster/api_server.py",
        "ui/app.py",
        "notebooks/adapter.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name == "engines" or name.startswith("engines.") for name in imported)


def test_notebook_async_stream_is_compatible(monkeypatch):
    import notebooks.adapter as adapter

    monkeypatch.setattr(adapter, "ensure_local_api_key", lambda: "test-key")

    async def fake_process_job(job):
        yield event_for(job.job_id)

    monkeypatch.setattr(adapter, "process_job", fake_process_job)
    result = asyncio.run(adapter.run_job("owner/model"))

    assert result.final_event.event_type is EventType.COMPLETE
