"""Asynchronous, metadata-only Hugging Face repository inspection."""

from __future__ import annotations

import os
import re
import logging
from typing import Any
from urllib.parse import quote

import aiohttp

from core.schemas import ModelMetaProfile


HF_ENDPOINT = "https://huggingface.co"
REQUEST_TIMEOUT_SECONDS = 30
INVENTORY_SUFFIXES = (".safetensors", ".bin", ".gguf", ".json")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")
SHARD_PATTERN = re.compile(r"-(\d+)-of-(\d+)\.[^.]+$", re.IGNORECASE)
LOGGER = logging.getLogger(__name__)


class HuggingFaceInspectionError(RuntimeError):
    """Base error for repository metadata inspection failures."""


class HuggingFaceAccessError(HuggingFaceInspectionError):
    """Raised when a gated or private repository cannot be inspected."""


def _request_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    headers = {"User-Agent": "haradibots/3.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _fetch_repo_metadata(
    repo_id: str,
    *,
    session: aiohttp.ClientSession,
) -> dict[str, Any] | None:
    encoded_repo = quote(repo_id.strip(), safe="/")
    if not encoded_repo or "/" not in encoded_repo:
        raise ValueError("repo_id must use the 'owner/name' form")

    url = f"{HF_ENDPOINT}/api/models/{encoded_repo}"
    try:
        async with session.get(
            url,
            params={"blobs": "true"},
            headers=_request_headers(),
        ) as response:
            if response.status == 404:
                return None
            if response.status in {401, 403}:
                raise HuggingFaceAccessError(
                    f"repository '{repo_id}' is gated or private; "
                    "set HF_TOKEN with an authorized token"
                )
            if response.status != 200:
                detail = (await response.text())[:200]
                raise HuggingFaceInspectionError(
                    f"Hugging Face metadata request failed with "
                    f"HTTP {response.status}: {detail}"
                )
            payload = await response.json(content_type=None)
    except TimeoutError as exc:
        raise HuggingFaceInspectionError(
            f"Hugging Face metadata request timed out for '{repo_id}'"
        ) from exc
    except aiohttp.ClientError as exc:
        raise HuggingFaceInspectionError(
            f"Hugging Face metadata request failed for '{repo_id}'"
        ) from exc

    if not isinstance(payload, dict):
        raise HuggingFaceInspectionError("Hugging Face returned malformed metadata")
    return payload


async def _fetch_json_file(
    repo_id: str,
    revision: str,
    filename: str,
    *,
    session: aiohttp.ClientSession,
) -> dict[str, Any] | None:
    encoded_repo = quote(repo_id.strip(), safe="/")
    encoded_revision = quote(revision, safe="")
    encoded_filename = quote(filename, safe="/")
    url = (
        f"{HF_ENDPOINT}/{encoded_repo}/resolve/"
        f"{encoded_revision}/{encoded_filename}"
    )
    try:
        async with session.get(
            url,
            headers=_request_headers(),
            allow_redirects=True,
        ) as response:
            if response.status == 404:
                return None
            if response.status in {401, 403}:
                raise HuggingFaceAccessError(
                    f"cannot read {filename} from gated or private "
                    f"repository '{repo_id}'; set an authorized HF_TOKEN"
                )
            if response.status != 200:
                raise HuggingFaceInspectionError(
                    f"failed to fetch {filename} from '{repo_id}': "
                    f"HTTP {response.status}"
                )
            if response.content_length is not None and response.content_length > 5_000_000:
                raise HuggingFaceInspectionError(
                    f"refusing oversized metadata file {filename}"
                )
            payload = await response.json(content_type=None)
    except TimeoutError as exc:
        raise HuggingFaceInspectionError(
            f"metadata file request timed out for '{repo_id}/{filename}'"
        ) from exc
    except aiohttp.ClientError as exc:
        raise HuggingFaceInspectionError(
            f"metadata file request failed for '{repo_id}/{filename}'"
        ) from exc

    return payload if isinstance(payload, dict) else None


def _first_int(config: dict[str, Any], *field_names: str) -> int | None:
    for field_name in field_names:
        value = config.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _classify_chat_template(
    tokenizer_config: dict[str, Any],
    model_type: str | None,
) -> str | None:
    template = tokenizer_config.get("chat_template")
    if template is None:
        return "gemma" if model_type in {"gemma", "gemma2", "gemma3"} else None
    template_text = str(template).lower()
    if "start_header_id" in template_text:
        return "llama3"
    if "im_start" in template_text:
        return "chatml"
    if "[inst]" in template_text:
        return "mistral"
    if "<|system|>" in template_text:
        return "phi3"
    if model_type in {"gemma", "gemma2", "gemma3"}:
        return "gemma"
    return "custom"


def _detect_quantization(
    config: dict[str, Any],
    filenames: list[str],
) -> tuple[bool, str | None]:
    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, dict):
        method = quantization_config.get(
            "quant_method",
            quantization_config.get("quantization_method"),
        )
        if isinstance(method, str) and method:
            return True, method.lower()

    lowered_names = [filename.lower() for filename in filenames]
    if any("awq" in filename for filename in lowered_names):
        return True, "awq"
    if any("gptq" in filename for filename in lowered_names):
        return True, "gptq"
    if any("exl2" in filename for filename in lowered_names):
        return True, "exl2"

    gguf_pattern = re.compile(
        r"(?:^|[._-])(q\d(?:_[a-z0-9]+)*|iq\d(?:_[a-z0-9]+)*)(?:[._-]|$)",
        re.IGNORECASE,
    )
    for filename in filenames:
        if not filename.lower().endswith(".gguf"):
            continue
        match = gguf_pattern.search(filename)
        if match:
            return True, match.group(1).upper()
        return True, "gguf"
    return False, None


def _quant_bits(config: dict[str, Any], quant_format: str | None) -> float | None:
    quantization = config.get("quantization_config")
    if isinstance(quantization, dict):
        for field in ("bits", "w_bit", "bit_width"):
            value = quantization.get(field)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    if quant_format:
        match = re.search(r"(\d+(?:\.\d+)?)", quant_format)
        if match:
            return float(match.group(1))
    return None


def _validated_profile(
    repo_id: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    required_model_fields = (
        "parameter_count",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "model_family",
    )
    for field_name in required_model_fields:
        if values.get(field_name) is None:
            LOGGER.warning(
                "repository '%s' is missing model metadata field '%s'",
                repo_id,
                field_name,
            )

    profile = ModelMetaProfile(repo_id=repo_id, **values)
    return profile.model_dump(mode="python")


async def inspect_repo(repo_id: str) -> dict[str, Any]:
    """Check repository existence, gating, and metadata-reported storage size."""

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        metadata = await _fetch_repo_metadata(repo_id, session=session)
        config = None
        tokenizer_config = None
        if metadata is not None:
            revision = metadata.get("sha")
            if not isinstance(revision, str) or not revision:
                revision = "main"
            try:
                config = await _fetch_json_file(
                    repo_id,
                    revision,
                    "config.json",
                    session=session,
                )
            except HuggingFaceAccessError as exc:
                LOGGER.warning("%s", exc)
            try:
                tokenizer_config = await _fetch_json_file(
                    repo_id,
                    revision,
                    "tokenizer_config.json",
                    session=session,
                )
            except HuggingFaceAccessError as exc:
                LOGGER.warning("%s", exc)

    if metadata is None:
        return _validated_profile(repo_id, {
            "repo_exists": False,
            "is_gated": False,
            "repo_size_bytes": 0,
            "parameter_count": None,
            "file_manifest": {},
            "num_shards": 0,
            "total_weight_bytes": 0,
            "num_layers": None,
            "hidden_size": None,
            "num_attention_heads": None,
            "num_key_value_heads": None,
            "vocab_size": None,
            "max_position_embeddings": None,
            "attention_type": "mha",
            "kv_head_ratio": None,
            "upper_bound_only": True,
            "chat_template_type": None,
            "is_prequantized": False,
            "quant_format": None,
            "quant_bits": None,
            "model_family": None,
        })

    siblings = metadata.get("siblings", [])
    sibling_size = sum(
        int(file_info.get("size") or 0)
        for file_info in siblings
        if isinstance(file_info, dict)
    )
    repo_size = metadata.get("usedStorage")
    if not isinstance(repo_size, int) or repo_size < 0:
        repo_size = sibling_size

    safetensors_metadata = metadata.get("safetensors")
    parameter_count = (
        safetensors_metadata.get("total")
        if isinstance(safetensors_metadata, dict)
        else None
    )
    if not isinstance(parameter_count, int) or parameter_count < 0:
        parameter_count = None

    file_manifest: dict[str, int] = {}
    num_shards = 0
    for file_info in siblings:
        if not isinstance(file_info, dict):
            continue
        filename = file_info.get("rfilename")
        if not isinstance(filename, str) or not filename.lower().endswith(
            INVENTORY_SUFFIXES
        ):
            continue
        size = file_info.get("size")
        if not isinstance(size, int):
            lfs = file_info.get("lfs")
            size = lfs.get("size", 0) if isinstance(lfs, dict) else 0
        file_manifest[filename] = max(int(size), 0)
        shard_match = SHARD_PATTERN.search(filename)
        if shard_match:
            num_shards = max(num_shards, int(shard_match.group(2)))

    config = config or {}
    tokenizer_config = tokenizer_config or {}
    num_layers = _first_int(
        config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    hidden_size = _first_int(config, "hidden_size", "n_embd", "d_model")
    num_attention_heads = _first_int(
        config,
        "num_attention_heads",
        "n_head",
    )
    configured_kv_heads = _first_int(config, "num_key_value_heads")
    upper_bound_only = configured_kv_heads is None
    num_key_value_heads = configured_kv_heads or num_attention_heads

    if num_attention_heads is None or num_attention_heads == 0:
        attention_type = "mha"
        kv_head_ratio = None
        upper_bound_only = True
    elif num_key_value_heads == 1:
        attention_type = "mqa"
        kv_head_ratio = 1.0 / num_attention_heads
    elif (
        num_key_value_heads is not None
        and num_key_value_heads < num_attention_heads
    ):
        attention_type = "gqa"
        kv_head_ratio = num_key_value_heads / num_attention_heads
    else:
        attention_type = "mha"
        kv_head_ratio = 1.0

    model_type_value = config.get("model_type")
    model_family = (
        model_type_value.lower()
        if isinstance(model_type_value, str) and model_type_value
        else None
    )
    is_prequantized, quant_format = _detect_quantization(
        config,
        list(file_manifest),
    )
    total_weight_bytes = sum(
        size
        for filename, size in file_manifest.items()
        if filename.lower().endswith(WEIGHT_SUFFIXES)
    )

    return _validated_profile(repo_id, {
        "repo_exists": True,
        "is_gated": bool(metadata.get("gated", False)),
        "repo_size_bytes": int(repo_size),
        "parameter_count": parameter_count,
        "file_manifest": file_manifest,
        "num_shards": num_shards,
        "total_weight_bytes": total_weight_bytes,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": _first_int(config, "vocab_size"),
        "max_position_embeddings": _first_int(
            config,
            "max_position_embeddings",
            "n_positions",
            "model_max_length",
        ),
        "attention_type": attention_type,
        "kv_head_ratio": kv_head_ratio,
        "upper_bound_only": upper_bound_only,
        "chat_template_type": _classify_chat_template(
            tokenizer_config,
            model_family,
        ),
        "is_prequantized": is_prequantized,
        "quant_format": quant_format,
        "quant_bits": _quant_bits(config, quant_format),
        "model_family": model_family,
    })
