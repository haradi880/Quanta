"""Low-frequency SQLite persistence for jobs and validation summaries."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_source: Mapped[str] = mapped_column(Text, nullable=False)
    output_format: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    state: Mapped[str] = mapped_column(String(32), nullable=False)


class ValidationResultRecord(Base):
    __tablename__ = "validation_results"

    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True
    )
    composite_delta: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    per_domain: Mapped[str] = mapped_column(Text, nullable=False)


def default_database_url() -> str:
    cache_root = Path(
        os.environ.get("HARADIBOTS_CACHE_ROOT", Path.home() / ".haradibots" / "cache")
    ).expanduser()
    db_dir = cache_root / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(db_dir / 'haradibots.sqlite3').as_posix()}"


def create_database(database_url: str | None = None):
    """Create the metadata database on first use and return its engine."""

    engine = create_engine(database_url or default_database_url())
    Base.metadata.create_all(engine)
    return engine


def insert_job(
    engine,
    *,
    job_id: str,
    model_source: str,
    output_format: str,
    state: str,
    started_at: datetime | None = None,
) -> Job:
    row = Job(
        job_id=job_id,
        model_source=model_source,
        output_format=output_format,
        state=state,
        started_at=started_at or datetime.now(timezone.utc),
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()
        session.refresh(row)
        session.expunge(row)
    return row


def upsert_job(
    engine,
    *,
    job_id: str,
    model_source: str,
    output_format: str,
    state: str,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> Job:
    """Create or update durable job metadata without duplicating a job ID."""

    with Session(engine) as session:
        row = session.get(Job, job_id)
        if row is None:
            row = Job(
                job_id=job_id,
                model_source=model_source,
                output_format=output_format,
                state=state,
                started_at=started_at or datetime.now(timezone.utc),
                completed_at=completed_at,
            )
            session.add(row)
        else:
            row.model_source = model_source
            row.output_format = output_format
            row.state = state
            row.completed_at = completed_at
        session.commit()
        session.refresh(row)
        session.expunge(row)
        return row


def get_validation_result(engine, job_id: str) -> dict[str, Any] | None:
    with Session(engine) as session:
        row = session.get(ValidationResultRecord, job_id)
        if row is None:
            return None
        return {
            "job_id": row.job_id,
            "composite_delta": row.composite_delta,
            "severity": row.severity,
            "per_domain": json.loads(row.per_domain),
        }


def get_job(engine, job_id: str) -> Job | None:
    with Session(engine) as session:
        row = session.get(Job, job_id)
        if row is not None:
            session.expunge(row)
        return row


def recover_incomplete_jobs(
    engine,
    *,
    recovered_at: datetime | None = None,
) -> list[str]:
    """Mark durable rows abandoned by a previous process as interrupted."""

    timestamp = recovered_at or datetime.now(timezone.utc)
    with Session(engine) as session:
        rows = list(
            session.scalars(
                select(Job).where(Job.completed_at.is_(None))
            )
        )
        recovered = [row.job_id for row in rows]
        for row in rows:
            row.state = "INTERRUPTED"
            row.completed_at = timestamp
        session.commit()
        return recovered


def insert_validation_result(
    engine,
    *,
    job_id: str,
    composite_delta: float,
    severity: str,
    per_domain: dict[str, Any],
) -> ValidationResultRecord:
    row = ValidationResultRecord(
        job_id=job_id,
        composite_delta=composite_delta,
        severity=severity,
        per_domain=json.dumps(per_domain, separators=(",", ":"), sort_keys=True),
    )
    with Session(engine) as session:
        session.merge(row)
        session.commit()
    return row
