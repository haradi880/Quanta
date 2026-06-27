"""Hardware-aware execution strategy planning."""

from __future__ import annotations

import re
from typing import Any

from core.profiler import calc_kv_cache, calc_weights_vram
from core.profiler import select_strategy as select_matrix_strategy
from core.schemas import StrategyConfig


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return value
    raise TypeError("hardware and model profiles must be dictionaries or Pydantic models")


def _parameter_count(model_meta: dict[str, Any]) -> float:
    value = model_meta.get("parameter_count")
    if value is None:
        size_billions = model_meta.get("model_size_b")
        value = (
            float(size_billions) * 1_000_000_000.0
            if size_billions is not None
            else None
        )
    if value is None or float(value) <= 0:
        raise ValueError("model metadata requires a positive parameter count")
    return float(value)


def _total_layers(model_meta: dict[str, Any]) -> int:
    value = model_meta.get("num_layers", model_meta.get("total_layers"))
    if value is None or int(value) <= 0:
        raise ValueError("model metadata requires a positive layer count")
    return int(value)


def _gpu_free_bytes(hw_profile: dict[str, Any]) -> list[int]:
    gpus = hw_profile.get("gpus")
    if isinstance(gpus, list):
        return [
            int(gpu.get("vram_free_bytes", gpu.get("vram_total_bytes", 0)))
            for gpu in gpus
            if isinstance(gpu, dict)
        ]
    value = hw_profile.get("vram_free_bytes", hw_profile.get("vram_total_bytes"))
    return [int(value)] if value is not None else []


def _bit_width(format_name: str) -> float:
    normalized = format_name.upper()
    if "FP32" in normalized:
        return 32.0
    if "FP16" in normalized or "BF16" in normalized:
        return 16.0
    if "AWQ" in normalized or "GPTQ" in normalized:
        return 4.0
    match = re.search(r"(?:^|[_/ -])(?:I?Q|INT)(\d+)", normalized)
    if match:
        return float(match.group(1))
    if "EXL2" in normalized:
        return 4.0
    raise ValueError(f"cannot determine bit width for format '{format_name}'")


def _parallelism(
    hardware_tier: str,
    hw_profile: dict[str, Any],
) -> tuple[int, int, int]:
    gpu_count = max(
        int(hw_profile.get("gpu_count", len(_gpu_free_bytes(hw_profile)))),
        1,
    )
    if hardware_tier == "Dual High VRAM":
        return 2, 1, 1
    if hardware_tier == "Multi-GPU Cluster":
        return gpu_count, 1, 1
    return 1, 1, 1


def _manual_vram_requirement(
    hw_profile: dict[str, Any],
    model_meta: dict[str, Any],
    target_format: str,
    gpu_layers: int,
) -> float:
    if gpu_layers <= 0:
        return 0.0

    parameters = _parameter_count(model_meta)
    total_layers = _total_layers(model_meta)
    if gpu_layers > total_layers:
        raise ValueError("manual gpu_layers cannot exceed model layer count")

    layer_fraction = float(gpu_layers) / float(total_layers)
    weights = calc_weights_vram(parameters, _bit_width(target_format)) * layer_fraction

    num_attention_heads = int(model_meta.get("num_attention_heads") or 0)
    hidden_size = int(model_meta.get("hidden_size") or 0)
    kv_heads = int(
        model_meta.get("num_key_value_heads") or num_attention_heads
    )
    context_length = int(model_meta.get("max_position_embeddings") or 2048)
    if num_attention_heads > 0 and hidden_size > 0 and kv_heads > 0:
        head_dimension = hidden_size / num_attention_heads
        kv_cache = calc_kv_cache(
            total_layers,
            1,
            context_length,
            kv_heads,
            head_dimension,
            2,
        ) * layer_fraction
    else:
        kv_cache = 0.0

    activation_buffer = 0.15 * weights
    gpu_count = max(len(_gpu_free_bytes(hw_profile)), 1)
    framework_overhead = (
        512.0 * 1024.0**2
        + max(gpu_count - 1, 0) * 128.0 * 1024.0**2
    )
    subtotal = weights + kv_cache + activation_buffer + framework_overhead
    safety_margin = max(0.08 * subtotal, 1.0 * 1024.0**3)
    return subtotal + safety_margin


def select_strategy(
    hw_profile: dict[str, Any] | Any,
    model_meta: dict[str, Any] | Any,
    *,
    mode: str = "auto",
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a concrete auto plan or a VRAM-checked manual override."""

    hardware = _as_dict(hw_profile)
    model = _as_dict(model_meta)
    matrix_result = select_matrix_strategy(hardware, model)
    total_layers = _total_layers(model)

    matrix_gpu_layers = matrix_result["gpu_layers"]
    gpu_layers = (
        total_layers
        if isinstance(matrix_gpu_layers, str)
        else int(matrix_gpu_layers)
    )
    target_format = str(matrix_result["recommended_format"])
    backend = str(matrix_result["backend"])
    tp_degree, pp_degree, dp_degree = _parallelism(
        str(matrix_result["hardware_tier"]),
        hardware,
    )

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"auto", "manual"}:
        raise ValueError("mode must be 'auto' or 'manual'")

    warning = False
    warning_reason = None
    if normalized_mode == "manual" and override:
        target_format = str(
            override.get("format", override.get("target_format", target_format))
        )
        gpu_layers = int(override.get("gpu_layers", gpu_layers))
        backend = str(override.get("backend", backend))
        tp_degree = int(override.get("tp_degree", tp_degree))
        pp_degree = int(override.get("pp_degree", pp_degree))
        dp_degree = int(override.get("dp_degree", dp_degree))

        required_vram = _manual_vram_requirement(
            hardware,
            model,
            target_format,
            gpu_layers,
        )
        available_vram = float(sum(_gpu_free_bytes(hardware)))
        if required_vram > available_vram:
            warning = True
            warning_reason = (
                f"manual override predicts {required_vram:.0f} bytes VRAM "
                f"against {available_vram:.0f} available bytes"
            )

    strategy = StrategyConfig(
        format=target_format,
        gpu_layers=gpu_layers,
        backend=backend,
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        dp_degree=dp_degree,
        warning=warning,
        warning_reason=warning_reason,
    )
    return strategy.model_dump(mode="python")


def check_overcompilation(
    source_format: str,
    target_format: str,
) -> dict[str, bool | str]:
    """Apply the Architecture §3.2 quantization conversion safety table."""

    source = source_format.upper().strip()
    target = target_format.upper().strip()
    if not source or not target:
        return {
            "allowed": False,
            "reason": "source and target formats are required",
        }

    if "AWQ" in source:
        return {
            "allowed": False,
            "reason": "AWQ matrices are calibrated, fused, and not safe to requantize",
        }
    if "GPTQ" in source:
        return {
            "allowed": False,
            "reason": "GPTQ quantization is not invertible",
        }
    if "EXL2" in source:
        return {
            "allowed": False,
            "reason": "EXL2 mixed-precision packing requires original calibration data",
        }

    source_bits = _bit_width(source)
    target_bits = _bit_width(target)
    if source_bits <= 4.0 and target_bits < source_bits:
        return {
            "allowed": False,
            "reason": "Q4_K_M or lower cannot be safely converted to a lower bit width",
        }

    full_precision = "FP16" in source or "BF16" in source
    if full_precision and target_bits <= 8.0:
        return {
            "allowed": True,
            "reason": "full-precision weights are a canonical quantization source",
        }
    if source_bits == 8.0 and target_bits in {6.0, 5.0, 4.0}:
        return {
            "allowed": True,
            "reason": "Q8 GGUF requantization is permitted with a loss warning",
        }
    if source_bits == 6.0 and target_bits == 4.0:
        return {
            "allowed": True,
            "reason": "Q6 to Q4_K_M is permitted with a loss warning",
        }
    return {
        "allowed": False,
        "reason": "conversion is not present in the approved safety table",
    }
