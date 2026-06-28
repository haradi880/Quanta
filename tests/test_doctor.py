import asyncio
from pathlib import Path

from core.runtime import NATIVE_ENV


def test_offline_doctor_checks_all_native_assets_and_redis(tmp_path, monkeypatch):
    import cli.doctor as doctor

    paths = {}
    for name, variable in NATIVE_ENV.items():
        filename = "convert_hf_to_gguf.py" if name.endswith(".py") else f"{name}.exe"
        path = tmp_path / filename
        path.write_bytes(b"asset")
        paths[name] = path
        monkeypatch.setenv(variable, str(path))
    (tmp_path / "conversion").mkdir()
    (tmp_path / "gguf-py").mkdir()
    probed = []

    async def probe(path, *arguments, **options):
        probed.append((Path(path).stem, arguments))
        return {"path": path, "version_output": "version"}

    class Manager:
        def __init__(self, binary_path, port):
            self.binary_path = binary_path
            self.port = port
            self.process = object()

        async def start(self, timeout_seconds):
            return f"redis://127.0.0.1:{self.port}/0"

        async def _reachable(self):
            return True

        async def stop(self):
            self.process = None

    monkeypatch.setattr(doctor, "configure_native_runtime", lambda: {})
    monkeypatch.setattr(doctor, "_probe_process", probe)
    monkeypatch.setattr(doctor, "LocalRedisManager", Manager)
    monkeypatch.setattr(
        doctor,
        "_verify_resp_operations",
        lambda url: asyncio.sleep(
            0,
            result={
                "PING": "PONG",
                "HSET": 2,
                "HGETALL": {"status": "healthy"},
                "SCAN": ["key"],
            },
        ),
    )

    result = asyncio.run(doctor.run_offline_doctor())

    assert result["status"] == "healthy"
    assert result["redis_stopped"] is True
    assert result["resp_checks"]["PING"] == "PONG"
    assert {name for name, _ in probed} == {
        "llama-cli",
        "llama-quantize",
        "llama-perplexity",
    }


def test_cli_doctor_returns_structured_success_and_failure(monkeypatch):
    import cli.main as cli

    async def healthy():
        return {"status": "healthy"}

    async def failed():
        raise RuntimeError("native failure")

    monkeypatch.setattr(cli, "run_offline_doctor", healthy)
    assert cli.main(["doctor", "--json"]) == 0

    monkeypatch.setattr(cli, "run_offline_doctor", failed)
    assert cli.main(["doctor", "--json"]) == 1
