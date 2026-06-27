"""Background Redis telemetry aggregation."""

from __future__ import annotations

import asyncio
import inspect
import logging
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
        sink: Callable[[list[dict[str, Any]]], Any] = push_to_prometheus,
    ) -> None:
        self.redis_client = redis_client or get_redis_client()
        self.interval_seconds = interval_seconds
        self.sink = sink
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
                if batch and (self._flush_task is None or self._flush_task.done()):
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
