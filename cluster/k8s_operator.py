"""Kubernetes Job generation and kubectl subprocess adapter."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


def generate_job_manifest(strategy_config: dict[str, Any]) -> str:
    job_id = str(strategy_config.get("job_id", "job")).lower().replace("_", "-")
    job_name = f"haradibots-{job_id}"[:63].rstrip("-")
    image = str(strategy_config.get("docker_image", "haradibots:cluster"))
    gpus = max(int(strategy_config.get("tp_degree", 1)), 1)
    payload = json.dumps(strategy_config, separators=(",", ":"))
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "labels": {"app": "haradibots"}},
        "spec": {
            "backoffLimit": 1,
            "template": {
                "metadata": {"labels": {"app": "haradibots"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "worker",
                            "image": image,
                            "command": [
                                "python3.11",
                                "-m",
                                "cluster.worker_entry",
                                "--scheduler",
                                "k8s",
                                "--job-id",
                                job_id,
                            ],
                            "env": [
                                {
                                    "name": "HARADIBOTS_STRATEGY_B64",
                                    "value": __import__("base64").b64encode(
                                        payload.encode()
                                    ).decode(),
                                }
                            ],
                            "resources": {
                                "limits": {"nvidia.com/gpu": gpus},
                                "requests": {"nvidia.com/gpu": gpus},
                            },
                            "volumeMounts": [
                                {
                                    "name": "mtls",
                                    "mountPath": "/run/haradibots/mtls",
                                    "readOnly": True,
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "mtls",
                            "secret": {"secretName": "haradibots-mtls"},
                        }
                    ],
                },
            },
        },
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def apply(manifest: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(manifest)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            ["kubectl", "apply", "--validate=true", "-f", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"kubectl apply failed: {result.stderr.strip()}")
        return result.stdout.strip()
    finally:
        path.unlink(missing_ok=True)


async def watch(job_id: str, timeout_seconds: float = 600) -> str:
    name = f"haradibots-{job_id.lower().replace('_', '-')}"[:63].rstrip("-")
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        process = await asyncio.create_subprocess_exec(
            "kubectl",
            "get",
            "job",
            name,
            "-o",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"kubectl get failed: {stderr.decode(errors='replace').strip()}"
            )
        status = json.loads(stdout).get("status", {})
        if int(status.get("succeeded", 0)) > 0:
            return "complete"
        if int(status.get("failed", 0)) > 0:
            return "failed"
        await asyncio.sleep(2)
    raise TimeoutError(f"Kubernetes job {name} did not finish")
