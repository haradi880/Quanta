"""Non-blocking Redis telemetry hot path."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from redis.asyncio import ConnectionPool, Redis

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_client: Redis | None = None


def get_redis_client() -> Redis:
    global _pool, _client
    if _client is None:
        _pool = ConnectionPool.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        _client = Redis(connection_pool=_pool)
    return _client


def _log_write_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("telemetry Redis write failed")


def write_tick(
    job_id: str,
    node_id: str,
    metrics_dict: dict[str, Any],
    *,
    client: Any | None = None,
) -> asyncio.Task[Any]:
    """Schedule one HSET and return immediately without awaiting Redis."""

    if not job_id or not node_id:
        raise ValueError("job_id and node_id are required")
    if not isinstance(metrics_dict, dict) or not metrics_dict:
        raise ValueError("metrics_dict must be a non-empty dictionary")
    redis_client = client or get_redis_client()
    task = asyncio.get_running_loop().create_task(
        redis_client.hset(f"telem:{job_id}:{node_id}", mapping=metrics_dict)
    )
    task.add_done_callback(_log_write_result)
    return task


async def close_redis_pool() -> None:
    global _pool, _client
    if _client is not None:
        await _client.aclose()
    if _pool is not None:
        await _pool.disconnect()
    _client = None
    _pool = None
