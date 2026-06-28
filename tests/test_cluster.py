import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient

from cluster.health import compute_parallelism, mtls_context, probe_node
from cluster.k8s_operator import apply, generate_job_manifest, watch
from cluster.ray_manager import _load_ray, collect_results, submit_job, terminate
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


def test_kubectl_apply_and_watch(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "job.batch/test configured\n", "")

    monkeypatch.setattr("cluster.k8s_operator.subprocess.run", fake_run)
    assert "configured" in apply(generate_job_manifest({"job_id": "test"}))

    class Process:
        returncode = 0

        async def communicate(self):
            return b'{"status":{"succeeded":1}}', b""

    async def fake_exec(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert asyncio.run(watch("test", timeout_seconds=1)) == "complete"


def test_node_server_returns_profiler_inventory(monkeypatch):
    import cluster.node_server as server

    monkeypatch.setattr(
        server,
        "snapshot",
        lambda: {"gpus": [{"uuid": "GPU-test"}]},
    )
    response = TestClient(server.app).get("/health/gpu")
    assert response.status_code == 200
    assert response.json()["gpus"][0]["uuid"] == "GPU-test"


def test_ray_submit_collect_and_terminate_with_scheduler_contract(monkeypatch):
    import cluster.ray_manager as manager

    killed = []

    class RemoteMethod:
        def __init__(self, function):
            self.function = function

        async def remote(self):
            return self.function()

    class ActorHandle:
        def __init__(self, actor):
            self.status = RemoteMethod(actor.status)
            self.terminate = RemoteMethod(actor.terminate)

    class ActorOptions:
        def __init__(self, cls):
            self.cls = cls

        def options(self, **kwargs):
            return self

        def remote(self, *args):
            return ActorHandle(self.cls(*args))

    class FakeRay:
        initialized = False

        def is_initialized(self):
            return self.initialized

        def init(self, **kwargs):
            self.initialized = True

        def get(self, value, timeout=None):
            return value

        def remote(self, **resources):
            return lambda cls: ActorOptions(cls)

        def kill(self, actor, no_restart):
            killed.append(actor)

    class Group:
        def ready(self):
            return True

    fake_ray = FakeRay()
    monkeypatch.setattr(
        manager,
        "_load_ray",
        lambda: (
            fake_ray,
            lambda bundles, strategy: Group(),
            lambda group: None,
            lambda **kwargs: kwargs,
        ),
    )
    handle = submit_job({"tp_degree": 2}, uuid4())
    events = asyncio.run(_collect_cluster(handle))
    assert len(events) == 2
    assert all(event.payload["status"] == "ready" for event in events)
    asyncio.run(terminate(handle))
    assert len(killed) == 2


@pytest.mark.integration
def test_real_local_ray_placement_collection_and_teardown(monkeypatch):
    ray = pytest.importorskip("ray")
    import cluster.ray_manager as manager

    monkeypatch.setenv("HARADIBOTS_RAY_ADDRESS", "local")
    handle = manager.submit_job(
        {
            "tp_degree": 2,
            "gpus_per_actor": 0,
            "cluster_test_cpu": True,
        },
        uuid4(),
    )
    try:
        events = asyncio.run(_collect_cluster(handle))
        assert [event.payload["shard_index"] for event in events] == [0, 1]
        assert all(event.payload["status"] == "ready" for event in events)
    finally:
        asyncio.run(manager.terminate(handle))
    assert ray.is_initialized() is False


async def _collect_cluster(handle):
    return [event async for event in collect_results(handle)]


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
