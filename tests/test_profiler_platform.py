import sys
from types import SimpleNamespace
from pathlib import Path

import core.profiler as profiler


def test_snapshot_gpu_collects_optional_nvml_fields(monkeypatch):
    memory = SimpleNamespace(total=16_000, free=12_000)
    remote = SimpleNamespace(busId=b"0000:01:00.0")
    fake = SimpleNamespace(
        NVMLError=RuntimeError,
        NVML_CLOCK_MEM=1,
        NVML_TEMPERATURE_GPU=2,
        NVML_NVLINK_MAX_LINKS=2,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: "gpu",
        nvmlDeviceGetMemoryInfo=lambda handle: memory,
        nvmlDeviceGetUUID=lambda handle: b"GPU-test",
        nvmlDeviceGetCudaComputeCapability=lambda handle: (7, 5),
        nvmlDeviceGetMemoryBusWidth=lambda handle: 256,
        nvmlDeviceGetMaxClockInfo=lambda handle, clock: 5000,
        nvmlDeviceGetNvLinkState=lambda handle, link: link == 0,
        nvmlDeviceGetNvLinkRemotePciInfo=lambda handle, link: remote,
        nvmlDeviceGetTemperature=lambda handle, sensor: 70,
        nvmlDeviceGetPowerUsage=lambda handle: 50_000,
        nvmlDeviceGetPowerManagementLimit=lambda handle: 70_000,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    result = profiler.snapshot_gpu()

    assert result == [
        {
            "uuid": "GPU-test",
            "vram_total_bytes": 16000,
            "vram_free_bytes": 12000,
            "cuda_cc_major": 7,
            "cuda_cc_minor": 5,
            "mem_bandwidth_gb_s": 320.0,
            "gpu_temp_c": 70.0,
            "power_draw_w": 50.0,
            "power_limit_w": 70.0,
            "nvlink_peers": ["0000:01:00.0"],
        }
    ]


def test_snapshot_gpu_falls_back_when_nvml_initialization_fails(monkeypatch):
    class NVMLFailure(Exception):
        pass

    fake = SimpleNamespace(
        NVMLError=NVMLFailure,
        nvmlInit=lambda: (_ for _ in ()).throw(NVMLFailure("driver")),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert profiler.snapshot_gpu() == []


def test_snapshot_cpu_hybrid_and_unknown_paths(monkeypatch):
    memory = SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3)
    frequencies = [
        SimpleNamespace(max=4000, current=3500),
        SimpleNamespace(max=3000, current=2500),
        SimpleNamespace(max=2000, current=1500),
    ]
    monkeypatch.setattr(profiler.psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(
        profiler.psutil,
        "cpu_count",
        lambda logical=False: 3 if not logical else 6,
    )
    monkeypatch.setattr(profiler.psutil, "cpu_freq", lambda percpu=True: frequencies)
    monkeypatch.setattr(profiler, "_windows_core_classes", lambda: ([0, 1], [2]))
    monkeypatch.setattr(profiler, "_read_isa_flags", lambda: ["AVX2"])

    hybrid = profiler.snapshot_cpu()
    assert hybrid["core_topology"] == "hybrid"
    assert hybrid["p_core_clock_ghz"] == 4.0
    assert hybrid["e_core_clock_ghz"] == 2.0

    monkeypatch.setattr(profiler, "_windows_core_classes", lambda: None)
    monkeypatch.setattr(profiler, "_linux_core_classes", lambda: None)
    unknown = profiler.snapshot_cpu()
    assert unknown["core_topology"] == "unknown"
    assert unknown["degraded_topology_detection"] is True


def test_snapshot_assembles_valid_profile(monkeypatch):
    monkeypatch.setattr(
        profiler,
        "snapshot_gpu",
        lambda: [
            {
                "uuid": "GPU-1",
                "vram_total_bytes": 100,
                "vram_free_bytes": 90,
                "cuda_cc_major": None,
                "cuda_cc_minor": None,
                "mem_bandwidth_gb_s": None,
                "gpu_temp_c": None,
                "power_draw_w": None,
                "power_limit_w": None,
                "nvlink_peers": [],
            }
        ],
    )
    monkeypatch.setattr(
        profiler,
        "snapshot_cpu",
        lambda: {
            "ram_total_gb": 16,
            "ram_available_gb": 8,
            "physical_cores": 4,
            "p_core_ids": [],
            "e_core_ids": [],
            "core_topology": "uniform",
            "p_core_clock_ghz": None,
            "e_core_clock_ghz": None,
            "isa_flags": [],
            "degraded_topology_detection": False,
        },
    )
    result = profiler.snapshot()
    assert result["gpu_count"] == 1
    assert result["gpu_uuids"] == ["GPU-1"]


def test_linux_core_classification_capacity_and_uniform(tmp_path, monkeypatch):
    monkeypatch.setattr(profiler.platform, "system", lambda: "Linux")
    root = tmp_path / "cpu"
    for cpu_id, capacity in ((0, 100), (1, 200), (2, 200)):
        cpu = root / f"cpu{cpu_id}"
        cpu.mkdir(parents=True)
        (cpu / "cpu_capacity").write_text(str(capacity), encoding="utf-8")

    original_path = Path

    def redirected_path(value):
        if str(value) == "/sys/devices/system/cpu":
            return root
        return original_path(value)

    monkeypatch.setattr(profiler, "Path", redirected_path)
    assert profiler._linux_core_classes() == ([1, 2], [0])

    (root / "cpu0" / "cpu_capacity").write_text("200", encoding="utf-8")
    assert profiler._linux_core_classes() == ([], [])


def test_isa_detection_linux_arm_and_windows(monkeypatch):
    class FakePath:
        def __init__(self, value):
            self.value = value

        def read_text(self, **kwargs):
            return "flags: avx2 avx512f asimd"

    monkeypatch.setattr(profiler, "Path", FakePath)
    monkeypatch.setattr(profiler.platform, "system", lambda: "Linux")
    assert profiler._read_isa_flags() == ["AVX-512", "AVX2", "NEON"]

    monkeypatch.setattr(profiler.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(profiler.platform, "machine", lambda: "arm64")
    assert profiler._read_isa_flags() == ["NEON"]


def test_profiler_matrix_validation_and_parameter_helpers(tmp_path):
    import json
    import pytest

    assert profiler._model_parameter_count({"parameter_count": 3}) == 3
    assert profiler._model_parameter_count({"model_size_b": 2}) == 2_000_000_000
    with pytest.raises(ValueError, match="must provide"):
        profiler._model_parameter_count({})
    assert profiler._gpu_free_bytes({"vram_free_bytes": 42}) == [42]

    invalid = tmp_path / "matrix.json"
    invalid.write_text(json.dumps({"tiers": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="every required tier"):
        profiler._load_decision_matrix(invalid)
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable or invalid"):
        profiler._load_decision_matrix(invalid)
