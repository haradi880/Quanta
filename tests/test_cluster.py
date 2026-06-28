import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from cluster.health import compute_parallelism, mtls_context, probe_node
from cluster.k8s_operator import generate_job_manifest
from cluster.ray_manager import _load_ray
from cluster.slurm_adapter import generate_batch_script, poll_status, submit


def test_mtls_is_mandatory_and_unreachable_probe_is_dead(tmp_path, monkeypatch):
    monkeypatch.setenv("HARADIBOTS_MTLS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="mTLS"):
        mtls_context("coordinator")
    start = time.monotonic()
    result = asyncio.run(
        probe_node(
            "dead-node",
            "127.0.0.1:1",
            client_node_id="coordinator",
            timeout_seconds=0.2,
        )
    )
    assert result["healthy"] is False
    assert time.monotonic() - start < 5


def test_asymmetric_parallelism_uses_only_healthy_gpus():
    nodes = [
        {"node_id": "n1", "healthy": True, "gpu_count": 2, "gpus": []},
        {"node_id": "n2", "healthy": True, "gpu_count": 1, "gpus": []},
        {
            "node_id": "n3",
            "healthy": False,
            "gpu_count": 1,
            "gpus": [{"uuid": "excluded-gpu"}],
        },
    ]
    plan = compute_parallelism(nodes, {"num_layers": 32})
    assert plan["tp_degree"] == 3
    assert plan["degraded"] is True
    assert plan["excluded_gpus"] == ["excluded-gpu"]
    assert plan["queue_job"] is False

    queued = compute_parallelism(nodes[1:], {"num_layers": 32})
    assert queued["queue_job"] is True


def test_slurm_generation_submission_and_status(monkeypatch, tmp_path):
    script = generate_batch_script({"tp_degree": 2, "model": "a'b"})
    assert "#SBATCH --gres=gpu:2" in script
    assert "HARADIBOTS_STRATEGY_B64" in script
    assert "a'b" not in script
    path = tmp_path / "job.sh"
    path.write_text(script, encoding="utf-8")

    def fake_run(command, **kwargs):
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(command, 0, "Submitted batch job 42\n", "")
        return subprocess.CompletedProcess(command, 0, "RUNNING\n", "")

    monkeypatch.setattr("cluster.slurm_adapter.subprocess.run", fake_run)
    assert submit(path) == "42"
    assert poll_status("42") == "RUNNING"


def test_kubernetes_manifest_has_gpu_and_mtls_mount():
    manifest = yaml.safe_load(
        generate_job_manifest(
            {
                "job_id": str(uuid4()),
                "tp_degree": 2,
                "docker_image": "haradibots:test",
            }
        )
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert container["volumeMounts"][0]["mountPath"] == "/run/haradibots/mtls"
    assert manifest["spec"]["template"]["spec"]["volumes"][0]["secret"][
        "secretName"
    ] == "haradibots-mtls"


def test_cluster_compose_has_no_worker_ports():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load(
        (root / "build" / "docker-compose.cluster.yml").read_text(encoding="utf-8")
    )
    assert compose["services"]["head"]["ports"]
    assert "ports" not in compose["services"]["worker-1"]
    assert "ports" not in compose["services"]["worker-2"]
    for service in compose["services"].values():
        assert service["environment"]["RAY_USE_TLS"] == "1"
    launcher = (root / "build" / "cluster_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "--ssl-cert-reqs 2" in launcher
    assert "cluster.node_server:app" in launcher


def test_ray_missing_dependency_has_clean_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "ray" or name.startswith("ray."):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="Ray cluster support"):
        _load_ray()


def test_certificate_script_issues_verifiable_node_cert(tmp_path):
    bash = shutil.which("bash")
    openssl = shutil.which("openssl")
    if not bash or not openssl:
        pytest.skip("bash/openssl are not on PATH")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["HARADIBOTS_MTLS_DIR"] = str(tmp_path)
    subprocess.run(
        [bash, str(root / "build" / "gen_certs.sh"), "--node", "test-node"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            openssl,
            "verify",
            "-CAfile",
            str(tmp_path / "ca.crt"),
            str(tmp_path / "test-node.crt"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
