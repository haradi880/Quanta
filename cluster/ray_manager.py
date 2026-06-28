"""Lazy Ray placement-group manager for optional cluster execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import UUID

from core.schemas import EventType, ProgressEvent


@dataclass
class RayJobHandle:
    job_id: UUID
    placement_group: Any
    actors: list[Any]
    ray: Any


def _load_ray():
    try:
        import ray
        from ray.util.placement_group import placement_group, remove_placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    except ImportError as exc:
        raise RuntimeError("Ray cluster support is not installed") from exc
    return ray, placement_group, remove_placement_group, PlacementGroupSchedulingStrategy


def submit_job(strategy_config: dict[str, Any], job_id: UUID) -> RayJobHandle:
    ray, placement_group, _remove_placement_group, scheduling_strategy = _load_ray()
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    shard_count = max(int(strategy_config.get("tp_degree", 1)), 1)
    bundles = [{"CPU": 1, "GPU": 1} for _ in range(shard_count)]
    group = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(group.ready())

    @ray.remote(num_cpus=1, num_gpus=1)
    class ModelShard:
        def __init__(self, shard_index, config):
            self.shard_index = shard_index
            self.config = config

        def status(self):
            return {
                "shard_index": self.shard_index,
                "status": "ready",
            }

        def terminate(self):
            return {"status": "terminated", "shard_index": self.shard_index}

    actors = [
        ModelShard.options(
            scheduling_strategy=scheduling_strategy(
                placement_group=group,
                placement_group_bundle_index=index,
            )
        ).remote(index, strategy_config)
        for index in range(shard_count)
    ]
    return RayJobHandle(
        job_id=job_id,
        placement_group=group,
        actors=actors,
        ray=ray,
    )


async def collect_results(
    handle: RayJobHandle,
) -> AsyncIterator[ProgressEvent]:
    for actor in handle.actors:
        status = await actor.status.remote()
        yield ProgressEvent(
            job_id=handle.job_id,
            event_type=EventType.CLUSTER_NODE_STATUS,
            timestamp_utc=datetime.now(timezone.utc),
            payload=status,
            telemetry={},
        )


async def terminate(handle: RayJobHandle) -> None:
    for actor in handle.actors:
        try:
            await actor.terminate.remote()
        finally:
            handle.ray.kill(actor, no_restart=True)
    try:
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(handle.placement_group)
    except ImportError:
        pass
