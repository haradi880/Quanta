"""Irreversible cache sanitization with architecture-mandated ordering."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import text

from telemetry.db import Base, create_database
from telemetry.redis_pipeline import close_redis_pool
from telemetry.redis_manager import stop_local_redis


PRISTINE_DIRECTORIES = (
    "auth",
    "db",
    "logs",
    "models",
    "output",
    "sources",
    "validation",
    "work",
)


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "HARADIBOTS_CACHE_ROOT",
            str(Path.home() / ".haradibots" / "cache"),
        )
    ).expanduser().resolve()


def validate_purge_root(root: Path) -> None:
    resolved = root.resolve()
    forbidden = {
        Path(resolved.anchor).resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValueError(f"refusing unsafe purge root: {resolved}")


async def sanitize_cache(
    teardown_all: Callable[[], Awaitable[None]],
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Execute teardown, DB drop/close, deletion, and pristine recreation."""

    target = (root or cache_root()).resolve()
    validate_purge_root(target)
    target.mkdir(parents=True, exist_ok=True)

    # Phase 1: harvest every process before touching persistent state.
    await teardown_all()

    # Phase 2: drop managed tables while SQLite is still open.
    (target / "db").mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{(target / 'db' / 'haradibots.sqlite3').as_posix()}"
    engine = await asyncio.to_thread(create_database, database_url)
    try:
        await asyncio.to_thread(Base.metadata.drop_all, engine)
        with engine.connect() as connection:
            connection.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
    finally:
        # Phase 3: release Redis and SQLite pools before filesystem deletion.
        await close_redis_pool()
        await stop_local_redis()
        await asyncio.to_thread(engine.dispose)

    # Phase 4: delete deepest-first without traversing external symlinks.
    failures: list[str] = []
    paths = sorted(
        target.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in paths:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        except OSError as exc:
            failures.append(f"{path}: {exc}")

    # Phase 5: recreate the known-empty local structure.
    for name in PRISTINE_DIRECTORIES:
        directory = target / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".keep").touch()
    if failures:
        log_path = target / "logs" / "manual_cleanup.log"
        log_path.write_text("\n".join(failures) + "\n", encoding="utf-8")

    return {
        "cache_root": str(target),
        "deleted_entries": len(paths) - len(failures),
        "failed_deletions": failures,
        "directories_recreated": list(PRISTINE_DIRECTORIES),
    }
