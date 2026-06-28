"""mTLS node probing and asymmetric healthy-GPU planning."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
from pathlib import Path
from typing import Any

import aiohttp


def _cluster_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "cluster.json"
    return json.loads(path.read_text(encoding="utf-8"))


def mtls_context(node_id: str) -> ssl.SSLContext:
    root = Path(
        os.environ.get(
            "HARADIBOTS_MTLS_DIR",
            Path(__file__).resolve().parent / "mtls",
        )
    )
    ca = root / "ca.crt"
    cert = root / f"{node_id}.crt"
    key = root / f"{node_id}.key"
    for path in (ca, cert, key):
        if not path.is_file():
            raise FileNotFoundError(f"required mTLS file is missing: {path}")
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(ca))
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = True
    return context


async def probe_node(
    node_id: str,
    addr: str,
    *,
    client_node_id: str = "coordinator",
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Return authenticated GPU inventory or a dead-node record within timeout."""

    timeout = timeout_seconds or float(
        _cluster_config()["probe_timeout_seconds"]
    )
    try:
        context = mtls_context(client_node_id)
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(
                f"https://{addr.rstrip('/')}/health/gpu",
                ssl=context,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"health endpoint returned {response.status}")
                payload = await response.json()
        if payload.get("hardware_fault"):
            raise RuntimeError(str(payload.get("hardware_fault")))
        gpus = payload.get("gpus")
        if not isinstance(gpus, list):
            raise RuntimeError("node returned invalid GPU inventory")
        return {
            "node_id": node_id,
            "addr": addr,
            "healthy": True,
            "gpu_count": len(gpus),
            "gpus": gpus,
            "error": None,
        }
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as exc:
        return {
            "node_id": node_id,
            "addr": addr,
            "healthy": False,
            "gpu_count": 0,
            "gpus": [],
            "error": str(exc),
        }


def compute_parallelism(
    healthy_nodes: list[dict[str, Any]],
    model_meta: dict[str, Any],
    *,
    min_healthy_threshold: int | None = None,
) -> dict[str, Any]:
    """Compute degrees from healthy GPUs only and report excluded inventory."""

    threshold = (
        min_healthy_threshold
        if min_healthy_threshold is not None
        else int(_cluster_config()["min_healthy_threshold"])
    )
    included = [node for node in healthy_nodes if node.get("healthy", True)]
    excluded = [node for node in healthy_nodes if not node.get("healthy", True)]
    total_gpus = sum(int(node.get("gpu_count", 0)) for node in included)
    excluded_gpus = [
        str(gpu.get("uuid", f"{node.get('node_id')}:gpu-{index}"))
        for node in excluded
        for index, gpu in enumerate(node.get("gpus", []))
    ]
    if total_gpus < threshold:
        return {
            "tp_degree": 0,
            "pp_degree": 0,
            "dp_degree": 0,
            "healthy_gpu_count": total_gpus,
            "excluded_gpus": excluded_gpus,
            "degraded": bool(excluded),
            "queue_job": True,
            "reason": f"healthy GPU count {total_gpus} is below threshold {threshold}",
        }
    layers = max(int(model_meta.get("num_layers") or total_gpus), 1)
    tp_degree = min(total_gpus, layers)
    return {
        "tp_degree": tp_degree,
        "pp_degree": 1,
        "dp_degree": 1,
        "healthy_gpu_count": total_gpus,
        "excluded_gpus": excluded_gpus,
        "degraded": bool(excluded),
        "queue_job": False,
        "reason": None,
    }
