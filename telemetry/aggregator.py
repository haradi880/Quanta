"""Background Redis telemetry aggregation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from prometheus_client import Gauge

from telemetry.redis_pipeline import get_redis_client

logger = logging.getLogger(__name__)
_PROMETHEUS_GAUGES: dict[str, Gauge] = {}


def _metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_:]", "_", f"haradibots_{name}")


def push_to_prometheus(batch: list[dict[str, Any]]) -> None:
    for row in batch:
        for metric, raw_value in row["metrics"].items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            gauge_name = _metric_name(metric)
            gauge = _PROMETHEUS_GAUGES.get(gauge_name)
            if gauge is None:
                gauge = Gauge(
                    gauge_name,
                    f"HaradiBots aggregated {metric}",
                    ("job_id", "node_id"),
                )
                _PROMETHEUS_GAUGES[gauge_name] = gauge
            gauge.labels(row["job_id"], row["node_id"]).set(value)


def push_to_postgresql(batch: list[dict[str, Any]]) -> None:
    """Persist aggregate snapshots only when server mode is explicitly enabled."""

    database_url = os.environ.get("POSTGRES_URL")
    if not database_url:
        raise RuntimeError("POSTGRES_URL is required for the PostgreSQL sink")
    import psycopg2

    with psycopg2.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    job_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    metrics_json JSONB NOT NULL
                )
                """
            )
            cursor.executemany(
                """
                INSERT INTO metrics (job_id, node_id, metrics_json)
                VALUES (%s, %s, %s::jsonb)
                """,
                [
                    (row["job_id"], row["node_id"], json.dumps(row["metrics"]))
                    for row in batch
                ],
            )


def configured_sink() -> Callable[[list[dict[str, Any]]], Any] | None:
    mode = os.environ.get("HARADIBOTS_TELEMETRY_SINK", "disabled").lower()
    if mode == "disabled":
        return None
    if mode == "prometheus":
        return push_to_prometheus
    if mode == "postgresql":
        return push_to_postgresql
    raise ValueError(
        "HARADIBOTS_TELEMETRY_SINK must be disabled, prometheus, or postgresql"
    )


async def collect_batch(redis_client: Any) -> list[dict[str, Any]]:
    batch: list[dict[str, Any]] = []
    async for key in redis_client.scan_iter(match="telem:*"):
        metrics = await redis_client.hgetall(key)
        _, job_id, node_id = key.split(":", 2)
        batch.append({"job_id": job_id, "node_id": node_id, "metrics": metrics})
    return batch


async def _flush(
    sink: Callable[[list[dict[str, Any]]], Any],
    batch: list[dict[str, Any]],
) -> None:
    try:
        if inspect.iscoroutinefunction(sink):
            await sink(batch)
        else:
            await asyncio.to_thread(sink, batch)
    except Exception:
        logger.exception("telemetry aggregate flush failed")


class TelemetryAggregator:
    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        interval_seconds: float = 10.0,
        sink: Callable[[list[dict[str, Any]]], Any] | None = None,
    ) -> None:
        self.redis_client = redis_client or get_redis_client()
        self.interval_seconds = interval_seconds
        self.sink = sink if sink is not None else configured_sink()
        self._task: asyncio.Task[None] | None = None
        self._flush_task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        return self._task

    async def _run(self) -> None:
        while True:
            try:
                batch = await collect_batch(self.redis_client)
                if (
                    batch
                    and self.sink is not None
                    and (self._flush_task is None or self._flush_task.done())
                ):
                    self._flush_task = asyncio.create_task(_flush(self.sink, batch))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("telemetry aggregation read failed")
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._flush_task is not None:
            await asyncio.gather(self._flush_task, return_exceptions=True)
            self._flush_task = None


def start_aggregator(**kwargs: Any) -> TelemetryAggregator:
    aggregator = TelemetryAggregator(**kwargs)
    aggregator.start()
    return aggregator
