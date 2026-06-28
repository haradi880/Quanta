"""Hardware-aware execution strategy planning."""

from __future__ import annotations

import math
import json
import asyncio
import re
import secrets
from pathlib import Path
from typing import Any

from core.profiler import calc_kv_cache, calc_weights_vram
from core.profiler import select_strategy as select_matrix_strategy
from core.schemas import (
    DomainValidationResult,
    GoldenValidationResult,
    StrategyConfig,
    ValidationResult,
)


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


def calc_perplexity(model: Any, token_sequence: Any) -> float:
    """Compute ``exp(-mean(log p(x_i)))`` from model likelihood output."""

    shape = getattr(token_sequence, "shape", None)
    if shape is not None and len(shape) > 0:
        sequence_length = int(shape[-1])
    else:
        try:
            sequence_length = len(token_sequence)
        except TypeError as exc:
            raise ValueError("token_sequence must be a sized sequence") from exc
    if sequence_length < 2:
        raise ValueError("perplexity requires at least two tokens")

    log_probability_method = getattr(model, "log_probabilities", None)
    if not callable(log_probability_method):
        log_probability_method = getattr(model, "log_likelihood", None)
    if callable(log_probability_method):
        likelihood = log_probability_method(token_sequence)
        if isinstance(likelihood, (int, float)):
            predicted_token_count = sequence_length - 1
            negative_mean_log_probability = (
                -float(likelihood) / predicted_token_count
            )
        else:
            log_probabilities = [float(value) for value in likelihood]
            if not log_probabilities:
                raise ValueError("model returned no log probabilities")
            negative_mean_log_probability = -sum(log_probabilities) / len(
                log_probabilities
            )
        return float(math.exp(negative_mean_log_probability))

    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise TypeError(
            "model must expose log probabilities or support a Torch forward pass"
        ) from exc

    input_ids = (
        token_sequence
        if isinstance(token_sequence, torch.Tensor)
        else torch.tensor(token_sequence, dtype=torch.long)
    )
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)

    loss = getattr(outputs, "loss", None)
    if loss is not None:
        return float(torch.exp(loss.detach()).item())

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise TypeError("model returned neither loss nor logits")
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    cross_entropy = torch_functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
    )
    return float(torch.exp(cross_entropy).item())


def generate_retrieval_prompt(
    context_length: int,
    model_max_ctx: int,
) -> str:
    """Generate an approximately token-sized document with one unique fact."""

    if context_length < 128:
        raise ValueError("context_length must be at least 128 tokens")
    if model_max_ctx < 128:
        raise ValueError("model_max_ctx must be at least 128 tokens")
    if context_length > model_max_ctx:
        raise ValueError("context_length cannot exceed model_max_ctx")

    project_id = f"HB-{secrets.token_hex(4).upper()}"
    verification_code = secrets.token_hex(8).upper()
    fact = (
        f" FACT: The verification code for project {project_id} "
        f"is {verification_code}."
    )
    question = (
        f"\nQUESTION: What is the verification code for project {project_id}?"
    )
    instruction = (
        "Read the archive carefully and answer the final question using only "
        "the document."
    )

    # Leading-space " archive" is one token in common GPT/BPE tokenizers and
    # remains close to one token in SentencePiece tokenizers.
    reserved_tokens = 48
    filler_count = max(context_length - reserved_tokens, 1)
    insertion_index = secrets.randbelow(filler_count + 1)
    filler_token = " archive"
    prefix = filler_token * insertion_index
    suffix = filler_token * (filler_count - insertion_index)
    return f"{instruction}\nDOCUMENT:{prefix}{fact}{suffix}{question}"


def _tokenize_validation_text(model: Any, text: str) -> Any:
    tokenize = getattr(model, "tokenize", None)
    if callable(tokenize):
        tokens = tokenize(text)
    else:
        tokenizer = getattr(model, "tokenizer", None)
        if callable(tokenizer):
            tokens = tokenizer(text, add_special_tokens=False)
        else:
            encode = getattr(model, "encode", None)
            if not callable(encode):
                raise TypeError("validation model must expose tokenize, tokenizer, or encode")
            tokens = encode(text)
    if isinstance(tokens, dict):
        tokens = tokens.get("input_ids")
    elif hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if tokens is None:
        raise ValueError("tokenizer returned no input_ids")
    return tokens


def _model_context_limit(model: Any) -> int:
    for owner in (model, getattr(model, "config", None)):
        if owner is None:
            continue
        for name in (
            "max_context_length",
            "max_position_embeddings",
            "n_positions",
            "model_max_length",
        ):
            value = getattr(owner, name, None)
            if isinstance(value, int) and 128 <= value < 1_000_000:
                return value
    tokenizer = getattr(model, "tokenizer", None)
    value = getattr(tokenizer, "model_max_length", None)
    return value if isinstance(value, int) and 128 <= value < 1_000_000 else 2048


def _mean_perplexity(model: Any, texts: list[str]) -> float:
    values = [
        calc_perplexity(model, _tokenize_validation_text(model, text))
        for text in texts
    ]
    if not values:
        raise ValueError("validation domain contains no prompts")
    return sum(values) / len(values)


def run_validation_suite(
    original_model: Any,
    quantized_model: Any,
    golden_prompts: list[Any] | None = None,
) -> ValidationResult:
    """Compare model perplexity across logic, retrieval, and code domains."""

    suite_path = Path(__file__).resolve().parents[1] / "config" / "validation_suite.json"
    with suite_path.open(encoding="utf-8") as suite_file:
        suite = json.load(suite_file)

    domain_texts: dict[str, list[str]] = {}
    for domain in ("logic", "code"):
        entries = suite.get(domain, [])
        domain_texts[domain] = [
            f"{entry['prompt']}\nReference answer:\n{entry['expected_output']}"
            for entry in entries
        ]

    max_context = min(
        _model_context_limit(original_model),
        _model_context_limit(quantized_model),
    )
    evaluated_context = min(max_context, 4096)
    domain_texts["retrieval"] = [
        generate_retrieval_prompt(evaluated_context, max_context)
    ]

    per_domain: dict[str, DomainValidationResult] = {}
    for domain, texts in domain_texts.items():
        original_perplexity = _mean_perplexity(original_model, texts)
        quantized_perplexity = _mean_perplexity(quantized_model, texts)
        delta = quantized_perplexity - original_perplexity
        per_domain[domain] = DomainValidationResult(
            original_perplexity=original_perplexity,
            quantized_perplexity=quantized_perplexity,
            delta=delta,
        )

    weights = suite["default_weights"]
    composite_delta = sum(
        per_domain[domain].delta * float(weights[domain])
        for domain in ("logic", "retrieval", "code")
    )
    severity = get_severity_tier(composite_delta)

    golden_results: list[GoldenValidationResult] = []
    for item in golden_prompts or []:
        if isinstance(item, str):
            prompt, expected = item, None
        elif hasattr(item, "model_dump"):
            data = item.model_dump(mode="python")
            prompt, expected = data["prompt"], data.get("expected_output")
        else:
            prompt, expected = item["prompt"], item.get("expected_output")
        scoring_text = (
            f"{prompt}\nReference answer:\n{expected}" if expected else prompt
        )
        original_ppl = _mean_perplexity(original_model, [scoring_text])
        quantized_ppl = _mean_perplexity(quantized_model, [scoring_text])
        golden_delta = quantized_ppl - original_ppl
        golden_results.append(
            GoldenValidationResult(
                prompt=prompt,
                expected_output=expected,
                actual_output=f"perplexity_delta={golden_delta:.6f}",
                passed=golden_delta <= 0.15,
            )
        )

    return ValidationResult(
        per_domain=per_domain,
        composite_delta=composite_delta,
        severity_tier=severity["severity_tier"],
        requires_confirmation=severity["requires_confirmation"],
        quarantined=severity["quarantined"],
        golden_results=golden_results,
    )


def get_severity_tier(composite_delta: float) -> dict[str, Any]:
    """Map weighted perplexity degradation to the architecture severity policy."""

    if not math.isfinite(composite_delta):
        raise ValueError("composite_delta must be finite")
    if composite_delta <= 0.05:
        tier = "excellent"
    elif composite_delta <= 0.15:
        tier = "good"
    elif composite_delta <= 0.35:
        tier = "moderate"
    elif composite_delta <= 0.60:
        tier = "poor"
    else:
        tier = "critical"
    return {
        "severity_tier": tier,
        "requires_confirmation": tier == "poor",
        "quarantined": tier == "critical",
    }


def _token_count_from_response(payload: dict[str, Any]) -> int:
    for key in ("count", "token_count", "num_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    raise ValueError("backend /tokenize response contains no token count")


def _offline_token_count(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        raise RuntimeError(
            "offline token estimation requires the model's Hugging Face tokenizer"
        )
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
    else:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("fallback tokenizer must be callable or expose encode()")
        encoded = encode(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded is None:
        raise ValueError("fallback tokenizer returned no input_ids")
    count = len(encoded)
    return int(math.ceil(count * 1.05))


async def count_tokens_native(
    text: str,
    backend_url: str | None,
    fallback_tokenizer: Any | None = None,
) -> int:
    """Use the live backend tokenizer, or HF tokenizer plus a 5% reserve."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if backend_url:
        import aiohttp

        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{backend_url.rstrip('/')}/tokenize",
                    json={"prompt": text},
                ) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            f"backend /tokenize returned HTTP {response.status}"
                        )
                    return _token_count_from_response(await response.json())
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            if fallback_tokenizer is None:
                raise
    return _offline_token_count(text, fallback_tokenizer)


def calc_available_ctx(
    max_ctx: int,
    system_prompt_tokens: int,
    history_tokens: int,
    online: bool,
) -> int | dict[str, Any]:
    """Return usable context tokens or a detailed overflow error."""

    if max_ctx < 1:
        raise ValueError("max_ctx must be positive")
    if system_prompt_tokens < 0 or history_tokens < 0:
        raise ValueError("token counts cannot be negative")
    used_tokens = system_prompt_tokens + history_tokens
    total_sequence_length = system_prompt_tokens + history_tokens
    safety_reserve = (
        0 if online else int(math.ceil(total_sequence_length * 0.05))
    )
    available_ctx = max_ctx - used_tokens - safety_reserve
    if available_ctx < 256:
        return {
            "error": "context_overflow_error",
            "max_ctx": max_ctx,
            "system_prompt_tokens": system_prompt_tokens,
            "history_tokens": history_tokens,
            "safety_reserve": safety_reserve,
            "available_ctx": available_ctx,
            "online": online,
        }
    return available_ctx
