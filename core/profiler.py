"""Deterministic hardware profiling and memory planning."""

from __future__ import annotations

import logging
import math
import os
import platform
import re
import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

import psutil

from core.schemas import HardwareProfile


LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")
DECISION_MATRIX_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "decision_matrix.json"
)


def _nvml_optional(
    operation: Callable[[], _T],
    default: _T | None = None,
) -> _T | None:
    """Return an optional NVML value without failing the whole snapshot."""

    try:
        return operation()
    except Exception:  # NVML exposes version-dependent exception subclasses.
        return default


def snapshot_gpu() -> list[dict[str, Any]]:
    """Return NVIDIA GPU telemetry, or an empty list when NVML is unavailable."""

    try:
        import pynvml
    except ImportError:
        LOGGER.info("pynvml is not installed; NVIDIA profiling is unavailable")
        return []

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        LOGGER.info("NVML initialization failed; using CPU fallback: %s", exc)
        return []

    devices: list[dict[str, Any]] = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            uuid_value = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid_value, bytes):
                uuid_value = uuid_value.decode("utf-8")

            compute_capability = _nvml_optional(
                lambda: pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            )
            if compute_capability is None:
                cc_major, cc_minor = None, None
            else:
                cc_major, cc_minor = compute_capability

            bus_width_bits = _nvml_optional(
                lambda: pynvml.nvmlDeviceGetMemoryBusWidth(handle)
            )
            memory_clock_mhz = _nvml_optional(
                lambda: pynvml.nvmlDeviceGetMaxClockInfo(
                    handle,
                    pynvml.NVML_CLOCK_MEM,
                )
            )
            bandwidth_gb_s = None
            if bus_width_bits is not None and memory_clock_mhz is not None:
                # Double-data-rate transfer: bytes/cycle × MHz × 2.
                bandwidth_gb_s = (
                    (float(bus_width_bits) / 8.0)
                    * float(memory_clock_mhz)
                    / 1000.0
                    * 2.0
                )

            nvlink_peers: list[str] = []
            max_links = getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 12)
            for link in range(max_links):
                is_active = _nvml_optional(
                    lambda link=link: pynvml.nvmlDeviceGetNvLinkState(handle, link),
                    False,
                )
                if not is_active:
                    continue
                remote_pci = _nvml_optional(
                    lambda link=link: pynvml.nvmlDeviceGetNvLinkRemotePciInfo(
                        handle,
                        link,
                    )
                )
                if remote_pci is None:
                    nvlink_peers.append(f"nvlink:{link}")
                    continue
                bus_id = getattr(remote_pci, "busId", f"nvlink:{link}")
                if isinstance(bus_id, bytes):
                    bus_id = bus_id.decode("utf-8")
                nvlink_peers.append(str(bus_id))

            devices.append(
                {
                    "uuid": str(uuid_value),
                    "vram_total_bytes": int(memory.total),
                    "vram_free_bytes": int(memory.free),
                    "cuda_cc_major": (
                        int(cc_major) if cc_major is not None else None
                    ),
                    "cuda_cc_minor": (
                        int(cc_minor) if cc_minor is not None else None
                    ),
                    "mem_bandwidth_gb_s": bandwidth_gb_s,
                    "gpu_temp_c": _nvml_optional(
                        lambda: float(
                            pynvml.nvmlDeviceGetTemperature(
                                handle,
                                pynvml.NVML_TEMPERATURE_GPU,
                            )
                        )
                    ),
                    "power_draw_w": _nvml_optional(
                        lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    ),
                    "power_limit_w": _nvml_optional(
                        lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                        / 1000.0
                    ),
                    "nvlink_peers": nvlink_peers,
                }
            )
    finally:
        pynvml.nvmlShutdown()

    return devices


def _windows_core_classes() -> tuple[list[int], list[int]] | None:
    """Read Windows CPU-set efficiency classes using the native topology API."""

    if os.name != "nt":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_cpu_sets = kernel32.GetSystemCpuSetInformation
        get_cpu_sets.argtypes = [
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
            wintypes.HANDLE,
            wintypes.ULONG,
        ]
        get_cpu_sets.restype = wintypes.BOOL

        required = wintypes.ULONG()
        get_cpu_sets(None, 0, ctypes.byref(required), None, 0)
        if required.value == 0:
            return None

        buffer = ctypes.create_string_buffer(required.value)
        if not get_cpu_sets(
            buffer,
            required.value,
            ctypes.byref(required),
            None,
            0,
        ):
            return None

        entries: list[tuple[int, int, int]] = []
        offset = 0
        while offset + 20 <= required.value:
            size = int.from_bytes(buffer.raw[offset : offset + 4], "little")
            info_type = int.from_bytes(buffer.raw[offset + 4 : offset + 8], "little")
            if size < 20 or offset + size > required.value:
                break
            if info_type == 0:  # CpuSetInformation
                logical_id = buffer.raw[offset + 14]
                core_index = buffer.raw[offset + 15]
                efficiency_class = buffer.raw[offset + 18]
                entries.append((logical_id, core_index, efficiency_class))
            offset += size

        if not entries:
            return None

        efficiency_classes = {entry[2] for entry in entries}
        if len(efficiency_classes) == 1:
            return [], []

        performance_class = max(efficiency_classes)
        p_core_ids: list[int] = []
        e_core_ids: list[int] = []
        seen_cores: set[tuple[int, int]] = set()
        for logical_id, core_index, efficiency_class in entries:
            core_key = (core_index, efficiency_class)
            if core_key in seen_cores:
                continue
            seen_cores.add(core_key)
            target = (
                p_core_ids
                if efficiency_class == performance_class
                else e_core_ids
            )
            target.append(logical_id)
        return sorted(p_core_ids), sorted(e_core_ids)
    except (AttributeError, OSError, ValueError):
        return None


def _linux_core_classes() -> tuple[list[int], list[int]] | None:
    """Classify Linux cores from sysfs capacity or maximum-frequency values."""

    if platform.system() != "Linux":
        return None

    cpu_root = Path("/sys/devices/system/cpu")
    measurements: dict[int, int] = {}
    for cpu_dir in cpu_root.glob("cpu[0-9]*"):
        match = re.fullmatch(r"cpu(\d+)", cpu_dir.name)
        if match is None:
            continue
        cpu_id = int(match.group(1))
        candidate_files = (
            cpu_dir / "cpu_capacity",
            cpu_dir / "cpufreq" / "cpuinfo_max_freq",
        )
        for candidate in candidate_files:
            try:
                measurements[cpu_id] = int(candidate.read_text().strip())
                break
            except (OSError, ValueError):
                continue

    if not measurements:
        return None
    classes = set(measurements.values())
    if len(classes) == 1:
        return [], []
    performance_value = max(classes)
    return (
        sorted(cpu_id for cpu_id, value in measurements.items() if value == performance_value),
        sorted(cpu_id for cpu_id, value in measurements.items() if value != performance_value),
    )


def _read_isa_flags() -> list[str]:
    machine = platform.machine().lower()
    detected: set[str] = set()

    if platform.system() == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8").lower()
        except OSError:
            cpuinfo = ""
        if re.search(r"\bavx2\b", cpuinfo):
            detected.add("AVX2")
        if re.search(r"\bavx512[a-z_]*\b", cpuinfo):
            detected.add("AVX-512")
        if re.search(r"\b(neon|asimd)\b", cpuinfo):
            detected.add("NEON")
    elif machine in {"arm64", "aarch64"}:
        detected.add("NEON")
    elif os.name == "nt":
        # PowerShell exposes the processor caption but not raw CPUID flags.
        # Keep the result conservative instead of claiming unsupported ISA.
        try:
            description = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | "
                    "Select-Object -First 1 -ExpandProperty Name)",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            ).stdout.lower()
        except (OSError, subprocess.SubprocessError):
            description = ""
        if "arm" in description:
            detected.add("NEON")

    return sorted(detected)


def snapshot_cpu() -> dict[str, Any]:
    """Return RAM, physical-core topology, clocks, and ISA capabilities."""

    memory = psutil.virtual_memory()
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores is None or physical_cores < 1:
        physical_cores = max((psutil.cpu_count(logical=True) or 2) - 1, 1)

    classified = _windows_core_classes()
    if classified is None:
        classified = _linux_core_classes()

    degraded_detection = classified is None
    if classified is None:
        p_core_ids: list[int] = []
        e_core_ids: list[int] = []
        topology = "unknown"
    else:
        p_core_ids, e_core_ids = classified
        topology = "hybrid" if p_core_ids and e_core_ids else "uniform"

    frequencies = psutil.cpu_freq(percpu=True) or []
    max_clocks = [
        float(freq.max or freq.current) / 1000.0
        for freq in frequencies
        if (freq.max or freq.current) > 0
    ]
    fallback_clock = max(max_clocks, default=0.0) or None

    def clock_for(core_ids: list[int]) -> float | None:
        values = [
            max_clocks[core_id]
            for core_id in core_ids
            if 0 <= core_id < len(max_clocks)
        ]
        return max(values, default=fallback_clock)

    return {
        "ram_total_gb": memory.total / (1024.0**3),
        "ram_available_gb": memory.available / (1024.0**3),
        "physical_cores": int(physical_cores),
        "p_core_ids": p_core_ids,
        "e_core_ids": e_core_ids,
        "core_topology": topology,
        "p_core_clock_ghz": clock_for(p_core_ids),
        "e_core_clock_ghz": clock_for(e_core_ids) if e_core_ids else None,
        "isa_flags": _read_isa_flags(),
        "degraded_topology_detection": degraded_detection,
    }


def get_thread_config(cpu_profile: dict[str, Any]) -> dict[str, Any]:
    """Build the Architecture §2.1 worker thread and pinning plan."""

    physical_cores = max(int(cpu_profile.get("physical_cores") or 1), 1)
    topology = cpu_profile.get("core_topology")
    p_core_ids = [int(core_id) for core_id in cpu_profile.get("p_core_ids", [])]

    if topology == "hybrid" and p_core_ids:
        return {
            "thread_count": len(p_core_ids),
            "core_ids": p_core_ids,
            "degraded_topology_detection": False,
        }

    degraded = topology not in {"hybrid", "uniform"} or bool(
        cpu_profile.get("degraded_topology_detection", False)
    )
    return {
        "thread_count": max(physical_cores - 1, 1),
        "core_ids": None,
        "degraded_topology_detection": degraded,
    }


def calc_weights_vram(params: float, bit_width: float) -> float:
    """Formula 1 weight memory: ``(bit_width / 8) × parameters`` bytes."""

    if params < 0 or bit_width <= 0:
        raise ValueError("params must be non-negative and bit_width must be positive")
    return (float(bit_width) / 8.0) * float(params)


def calc_kv_cache(
    L: float,
    B: float,
    N: float,
    H_kv: float,
    D: float,
    P_kv: float,
) -> float:
    """Formula 2 GQA-aware Key+Value cache footprint in bytes."""

    terms = (L, B, N, H_kv, D, P_kv)
    if any(term < 0 for term in terms):
        raise ValueError("KV-cache terms must be non-negative")
    return 2.0 * float(L) * float(B) * float(N) * float(H_kv) * float(D) * float(P_kv)


def calc_partial_offload_layers(
    vram_free: float,
    overhead: float,
    weights_vram: float,
    total_layers: int,
) -> int:
    """Formula 3 maximum number of whole transformer layers fitting in VRAM."""

    if min(vram_free, overhead, weights_vram) < 0:
        raise ValueError("VRAM values must be non-negative")
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if weights_vram == 0:
        return total_layers

    bytes_per_layer = float(weights_vram) / float(total_layers)
    available = max(float(vram_free) - float(overhead), 0.0)
    fitting_layers = math.floor(available / bytes_per_layer)
    return min(max(fitting_layers, 0), total_layers)


def _model_parameter_count(model_meta: dict[str, Any]) -> float:
    for field_name in ("parameter_count", "num_parameters", "params"):
        value = model_meta.get(field_name)
        if value is not None:
            parameter_count = float(value)
            if parameter_count <= 0:
                raise ValueError("model parameter count must be positive")
            return parameter_count
    size_billions = model_meta.get("model_size_b")
    if size_billions is not None:
        parameter_count = float(size_billions) * 1_000_000_000.0
        if parameter_count <= 0:
            raise ValueError("model size must be positive")
        return parameter_count
    raise ValueError("model_meta must provide parameter_count or model_size_b")


def _gpu_free_bytes(hw_profile: dict[str, Any]) -> list[int]:
    gpus = hw_profile.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        return [
            int(gpu.get("vram_free_bytes", gpu.get("vram_total_bytes", 0)))
            for gpu in gpus
            if isinstance(gpu, dict)
        ]
    value = hw_profile.get("vram_free_bytes", hw_profile.get("vram_total_bytes"))
    return [int(value)] if value is not None else []


def _load_decision_matrix(
    path: Path | str = DECISION_MATRIX_PATH,
) -> dict[str, dict[str, Any]]:
    try:
        matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("decision matrix is unavailable or invalid") from exc
    tiers = matrix.get("tiers") if isinstance(matrix, dict) else None
    if not isinstance(tiers, list):
        raise ValueError("decision matrix must contain a tiers array")
    indexed = {
        tier["id"]: tier
        for tier in tiers
        if isinstance(tier, dict) and isinstance(tier.get("id"), str)
    }
    required = {
        "cpu_x86",
        "cpu_arm",
        "apple_silicon",
        "low_vram_gpu",
        "mid_vram_gpu",
        "high_vram_gpu",
        "dual_high_vram",
        "multi_gpu_cluster",
    }
    if indexed.keys() != required:
        raise ValueError("decision matrix does not contain every required tier")
    return indexed


def select_strategy(
    hw_profile: dict[str, Any],
    model_meta: dict[str, Any],
) -> dict[str, Any]:
    """Select the deterministic Architecture §2.3 recommendation."""

    tiers = _load_decision_matrix()
    parameters = _model_parameter_count(model_meta)
    model_size_b = parameters / 1_000_000_000.0
    free_by_gpu = _gpu_free_bytes(hw_profile)
    gpu_count = int(hw_profile.get("gpu_count", len(free_by_gpu)))
    architecture = str(
        hw_profile.get("architecture", platform.machine())
    ).lower()
    system_name = str(
        hw_profile.get("platform", platform.system())
    ).lower()

    if hw_profile.get("cluster_config") is not None or gpu_count > 2:
        tier_id = "multi_gpu_cluster"
        if model_size_b < 70:
            LOGGER.info("cluster tier selected for a sub-70B model")
    elif gpu_count == 2 and min(free_by_gpu or [0]) >= 20 * 1024**3:
        tier_id = "dual_high_vram"
    elif gpu_count >= 1:
        max_free = max(free_by_gpu or [0])
        free_gib = max_free / (1024.0**3)
        if free_gib >= 20:
            tier_id = "high_vram_gpu"
        elif free_gib >= 10:
            tier_id = "mid_vram_gpu"
        elif free_gib >= 3.5:
            tier_id = "low_vram_gpu"
        else:
            tier_id = "cpu_arm" if "arm" in architecture else "cpu_x86"
    elif system_name == "darwin" and architecture in {"arm64", "aarch64"}:
        tier_id = "apple_silicon"
    elif architecture in {"arm64", "aarch64", "arm"}:
        tier_id = "cpu_arm"
    else:
        tier_id = "cpu_x86"

    tier = tiers[tier_id]
    max_model_size_b = tier.get("max_model_size_b")
    if max_model_size_b is not None and model_size_b > float(max_model_size_b):
        raise ValueError(
            f"{model_size_b:g}B model exceeds the "
            f"{tier['hardware_tier']} matrix limit of {max_model_size_b:g}B"
        )
    gpu_layers: int | str = tier["gpu_layers"]
    if gpu_layers == "calculated":
        total_layers = int(
            model_meta.get("num_layers", model_meta.get("total_layers", 1))
        )
        overhead = 512.0 * 1024.0**2 + max(gpu_count - 1, 0) * 128.0 * 1024.0**2
        weights = calc_weights_vram(parameters, 4)
        gpu_layers = calc_partial_offload_layers(
            max(free_by_gpu or [0]),
            overhead,
            weights,
            total_layers,
        )

    return {
        "hardware_tier": tier["hardware_tier"],
        "recommended_format": tier["recommended_format"],
        "gpu_layers": gpu_layers,
        "backend": tier["backend"],
    }


def snapshot() -> dict[str, Any]:
    """Assemble and validate a complete version 3.0 hardware profile."""

    gpu_profiles = snapshot_gpu()
    cpu_profile = snapshot_cpu()
    profile = HardwareProfile(
        profile_id=uuid4(),
        timestamp_utc=datetime.now(timezone.utc),
        gpu_count=len(gpu_profiles),
        gpu_uuids=[gpu["uuid"] for gpu in gpu_profiles],
        gpus=gpu_profiles,
        cpu=cpu_profile,
    )
    return profile.model_dump(mode="python")
