import asyncio
import json
import time
from pathlib import Path

import pytest
from sqlalchemy import inspect

from telemetry.aggregator import (
    TelemetryAggregator,
    _flush,
    collect_batch,
    configured_sink,
    push_to_prometheus,
)
from telemetry.db import create_database, get_job, insert_job
from telemetry.redis_pipeline import write_tick
from telemetry.redis_manager import LocalRedisManager
from telemetry.warnings import evaluate_tick


class FakeRedis:
    def __init__(self):
        self.data = {"telem:job-1:node-1": {"vram_pct": "50"}}
        self.writes = []

    async def scan_iter(self, match):
        for key in self.data:
            yield key

    async def hgetall(self, key):
        return self.data[key]

    async def hset(self, key, mapping):
        self.writes.append((key, mapping))


def test_sqlite_stores_metadata_without_telemetry_ticks(tmp_path):
    engine = create_database(f"sqlite:///{(tmp_path / 'meta.sqlite').as_posix()}")
    insert_job(
        engine,
        job_id="job-1",
        model_source="test/model",
        output_format="GGUF",
        state="RUNNING",
    )

    assert get_job(engine, "job-1").model_source == "test/model"
    assert inspect(engine).get_table_names() == ["jobs", "validation_results"]
    engine.dispose()


def test_write_tick_returns_without_waiting_for_redis():
    async def scenario():
        class SlowRedis(FakeRedis):
            async def hset(self, key, mapping):
                await asyncio.sleep(0.05)
                await super().hset(key, mapping)

        redis = SlowRedis()
        start = time.perf_counter()
        task = write_tick("job-1", "node-1", {"vram_pct": 50}, client=redis)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.001
        assert not task.done()
        await task

    asyncio.run(scenario())


def test_slow_aggregator_sink_does_not_block_hot_path():
    async def scenario():
        redis = FakeRedis()
        sink_started = asyncio.Event()

        async def slow_sink(batch):
            sink_started.set()
            await asyncio.sleep(0.1)

        aggregator = TelemetryAggregator(
            redis_client=redis,
            interval_seconds=0.01,
            sink=slow_sink,
        )
        aggregator.start()
        await asyncio.wait_for(sink_started.wait(), timeout=0.2)
        start = time.perf_counter()
        task = write_tick("job-2", "node-2", {"cpu_pct": 20}, client=redis)
        assert time.perf_counter() - start < 0.001
        await task
        await aggregator.stop()
        assert redis.writes

    asyncio.run(scenario())


def test_standalone_aggregator_has_no_durable_sink_by_default(monkeypatch):
    monkeypatch.delenv("HARADIBOTS_TELEMETRY_SINK", raising=False)

    assert configured_sink() is None


def test_aggregator_collection_prometheus_and_sink_modes(monkeypatch):
    redis = FakeRedis()
    batch = asyncio.run(collect_batch(redis))
    assert batch == [
        {
            "job_id": "job-1",
            "node_id": "node-1",
            "metrics": {"vram_pct": "50"},
        }
    ]
    push_to_prometheus(batch)

    monkeypatch.setenv("HARADIBOTS_TELEMETRY_SINK", "prometheus")
    assert configured_sink() is push_to_prometheus
    monkeypatch.setenv("HARADIBOTS_TELEMETRY_SINK", "postgresql")
    assert configured_sink().__name__ == "push_to_postgresql"
    monkeypatch.setenv("HARADIBOTS_TELEMETRY_SINK", "invalid")
    with pytest.raises(ValueError, match="must be"):
        configured_sink()


def test_flush_supports_sync_async_and_contains_sink_failures():
    calls = []

    def sync_sink(batch):
        calls.append(("sync", batch))

    async def async_sink(batch):
        calls.append(("async", batch))

    async def broken(batch):
        raise RuntimeError("sink down")

    asyncio.run(_flush(sync_sink, [{"x": 1}]))
    asyncio.run(_flush(async_sink, [{"x": 2}]))
    asyncio.run(_flush(broken, [{"x": 3}]))
    assert calls == [
        ("sync", [{"x": 1}]),
        ("async", [{"x": 2}]),
    ]


def test_threshold_file_contains_five_complete_metrics():
    path = Path(__file__).resolve().parents[1] / "config" / "telemetry_thresholds.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]

    assert len(metrics) == 5
    for policy in metrics.values():
        assert {"warning", "critical", "emergency"} <= policy.keys()
        for level in ("warning", "critical", "emergency"):
            assert "system_action" in policy[level]


def test_vram_emergency_aborts():
    alerts = evaluate_tick({"vram_pct": 98})

    assert len(alerts) == 1
    assert alerts[0].level == "emergency"
    assert alerts[0].system_action == "abort"


def test_local_redis_reuses_reachable_loopback_process():
    async def scenario():
        async def handler(reader, writer):
            await reader.read(64)
            writer.write(b"+PONG\r\n")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        manager = LocalRedisManager(port=port)
        try:
            url = await manager.start()
            assert url == f"redis://127.0.0.1:{port}/0"
            assert manager.owns_process is False
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_local_redis_requires_bundled_binary(monkeypatch, tmp_path):
    import telemetry.redis_manager as module

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    manager = LocalRedisManager(
        binary_path=str(tmp_path / "missing-redis"),
        port=6398,
    )

    with pytest.raises(FileNotFoundError, match="bundled Garnet"):
        asyncio.run(manager.start(timeout_seconds=0.1))
