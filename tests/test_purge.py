import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import inspect

from core.orchestrator import Orchestrator
from core.purge import PRISTINE_DIRECTORIES, sanitize_cache, validate_purge_root
from telemetry.db import create_database, insert_job


def test_purge_orders_teardown_before_db_and_filesystem_cleanup(tmp_path):
    root = tmp_path / "sandbox" / "cache"
    db_path = root / "db" / "haradibots.sqlite3"
    db_path.parent.mkdir(parents=True)
    engine = create_database(f"sqlite:///{db_path.as_posix()}")
    insert_job(
        engine,
        job_id="purge-test",
        model_source="owner/model",
        output_format="GGUF",
        state="RUNNING",
    )
    engine.dispose()
    artifact = root / "models" / "artifact.gguf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"locked candidate")
    phases = []

    async def teardown():
        phases.append("teardown")
        check = create_database(f"sqlite:///{db_path.as_posix()}")
        try:
            assert inspect(check).get_table_names() == ["jobs", "validation_results"]
        finally:
            check.dispose()
        assert artifact.exists()

    result = asyncio.run(sanitize_cache(teardown, root=root))

    assert phases == ["teardown"]
    assert result["failed_deletions"] == []
    assert not db_path.exists()
    for name in PRISTINE_DIRECTORIES:
        assert (root / name / ".keep").is_file()
    assert not artifact.exists()


def test_purge_rejects_dangerous_roots():
    with pytest.raises(ValueError, match="unsafe purge root"):
        validate_purge_root(Path.home())


def test_orchestrator_purge_harvests_registered_workers(tmp_path, monkeypatch):
    root = tmp_path / "runtime" / "cache"
    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(root))
    terminated = []

    class Worker:
        process = None

        async def terminate(self):
            terminated.append(True)

        def is_alive(self):
            return False

    orchestrator = Orchestrator(teardown_grace_seconds=0)
    job_id = uuid4()
    orchestrator.register_worker(job_id, Worker())

    asyncio.run(orchestrator.purge())

    assert terminated == [True]
    assert orchestrator.worker_handles(job_id) == ()
