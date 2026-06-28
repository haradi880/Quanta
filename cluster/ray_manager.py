"""Lazy Ray placement-group manager for optional cluster execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, AsyncIterator
from uuid import UUID

from core.schemas import EventType, ProgressEvent


@dataclass
class RayJobHandle:
    job_id: UUID
    placement_group: Any
    actors: list[Any]
    ray: Any
    remove_placement_group: Any
    owns_runtime: bool = False


def _load_ray():
    try:
        import ray
        from ray.util.placement_group import placement_group, remove_placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    except ImportError as exc:
        raise RuntimeError("Ray cluster support is not installed") from exc
    return ray, placement_group, remove_placement_group, PlacementGroupSchedulingStrategy


def submit_job(strategy_config: dict[str, Any], job_id: UUID) -> RayJobHandle:
    ray, placement_group, remove_placement_group, scheduling_strategy = _load_ray()
    owns_runtime = False
    if not ray.is_initialized():
        address = os.environ.get("HARADIBOTS_RAY_ADDRESS", "auto")
        if address == "local":
            ray.init(
                include_dashboard=False,
                ignore_reinit_error=True,
            )
            owns_runtime = True
        else:
            ray.init(address=address, ignore_reinit_error=True)
    shard_count = max(int(strategy_config.get("tp_degree", 1)), 1)
    gpu_per_actor = float(strategy_config.get("gpus_per_actor", 1))
    if gpu_per_actor <= 0 and not strategy_config.get("cluster_test_cpu", False):
        raise ValueError("production Ray actors require a positive GPU allocation")
    bundle = {"CPU": 1}
    if gpu_per_actor > 0:
        bundle["GPU"] = gpu_per_actor
    bundles = [dict(bundle) for _ in range(shard_count)]
    placement_strategy = (
        "STRICT_PACK"
        if strategy_config.get("cluster_test_cpu", False)
        else "STRICT_SPREAD"
    )
    group = placement_group(bundles, strategy=placement_strategy)
    try:
        ray.get(
            group.ready(),
            timeout=float(strategy_config.get("placement_timeout_seconds", 30)),
        )
    except Exception as exc:
        remove_placement_group(group)
        if owns_runtime:
            ray.shutdown()
        raise RuntimeError("Ray placement group did not become ready") from exc

    @ray.remote(num_cpus=1, num_gpus=max(gpu_per_actor, 0))
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
        remove_placement_group=remove_placement_group,
        owns_runtime=owns_runtime,
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
    handle.remove_placement_group(handle.placement_group)
    if handle.owns_runtime:
        handle.ray.shutdown()
