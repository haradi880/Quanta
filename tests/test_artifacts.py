import struct
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.artifacts import (
    ArtifactCompatibilityError,
    inspect_gguf_metadata,
    select_gguf_file,
)
from core.hf_inspector import _estimate_parameter_count
from core.orchestrator import Orchestrator
from core.schemas import (
    AuthBlock,
    CallbackConfig,
    EventType,
    InterfaceType,
    JobEnvelope,
    JobMode,
    ModelSource,
    ProgressEvent,
    SystemPrompt,
)


def _gguf_string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_entry(key, value_type, value):
    data = _gguf_string(key) + struct.pack("<I", value_type)
    if value_type == 4:
        return data + struct.pack("<I", value)
    if value_type == 8:
        return data + _gguf_string(value)
    if value_type == 9:
        values = value
        return (
            data
            + struct.pack("<I", 8)
            + struct.pack("<Q", len(values))
            + b"".join(_gguf_string(item) for item in values)
        )
    raise AssertionError("unsupported test metadata type")


def test_gguf_selection_prefers_target_and_excludes_mmproj():
    manifest = {
        "model-Q8_0.gguf": 200,
        "model-Q4_K_M.gguf": 100,
        "mmproj-Q4_K_M.gguf": 10,
    }

    assert select_gguf_file(manifest, "Q4_K_M") == "model-Q4_K_M.gguf"


def test_non_gguf_repository_is_rejected_before_worker_launch():
    with pytest.raises(ArtifactCompatibilityError, match="requires a GGUF"):
        select_gguf_file({"model.safetensors": 100}, "Q4_K_M")


def test_gguf_header_supplies_planning_metadata(tmp_path):
    entries = [
        _metadata_entry("general.architecture", 8, "llama"),
        _metadata_entry("llama.block_count", 4, 22),
        _metadata_entry("llama.embedding_length", 4, 2048),
        _metadata_entry("llama.attention.head_count", 4, 32),
        _metadata_entry("llama.attention.head_count_kv", 4, 4),
        _metadata_entry("llama.context_length", 4, 4096),
        _metadata_entry("tokenizer.ggml.tokens", 9, ["a", "b", "c"]),
    ]
    path = tmp_path / "model.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(entries))
        + b"".join(entries)
    )

    metadata = inspect_gguf_metadata(path)

    assert metadata == {
        "model_family": "llama",
        "num_layers": 22,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "vocab_size": 3,
    }


def test_weight_size_parameter_fallback_is_conservative():
    assert _estimate_parameter_count({"pytorch_model.bin": 4000}, None) == 1000
    assert _estimate_parameter_count({"model-Q4_0.gguf": 500}, 4) == 1000


def test_orchestrator_passes_acquired_gguf_path_to_worker(tmp_path, monkeypatch):
    import core.orchestrator as module

    model_path = tmp_path / "model-Q4_0.gguf"
    model_path.write_bytes(b"GGUF")
    job_id = uuid4()
    captured = {}

    class Worker:
        process = None

        async def execute(self, strategy):
            captured.update(strategy)
            yield ProgressEvent(
                schema_version="3.0",
                job_id=job_id,
                event_type=EventType.QUANTIZATION_PROGRESS,
                timestamp_utc=datetime.now(timezone.utc),
                payload={"status": "complete"},
                telemetry={},
            )

        async def validate(self, prompts):
            domain = {
                "original_perplexity": 10.0,
                "quantized_perplexity": 10.01,
                "delta": 0.01,
            }
            return {
                "per_domain": {
                    "logic": domain,
                    "retrieval": domain,
                    "code": domain,
                },
                "composite_delta": 0.01,
                "severity_tier": "excellent",
                "requires_confirmation": False,
                "quarantined": False,
                "golden_results": [],
            }

        async def terminate(self):
            return None

    monkeypatch.setattr(module, "authenticate", lambda envelope: {"subject": "test"})
    monkeypatch.setattr(
        module,
        "snapshot",
        lambda: {
            "profile_id": uuid4(),
            "timestamp_utc": datetime.now(timezone.utc),
            "gpu_count": 0,
            "gpu_uuids": [],
            "gpus": [],
            "cpu": {
                "ram_total_gb": 16,
                "ram_available_gb": 12,
                "physical_cores": 4,
                "p_core_ids": [],
                "e_core_ids": [],
                "core_topology": "unknown",
                "p_core_clock_ghz": None,
                "e_core_clock_ghz": None,
                "isa_flags": ["AVX2"],
                "degraded_topology_detection": True,
            },
        },
    )
    monkeypatch.setattr(
        module,
        "inspect_repo",
        lambda repo_id: _async_value(
            {
                "repo_exists": True,
                "parameter_count": 15_000_000,
                "file_manifest": {"model-Q4_0.gguf": 10},
                "num_layers": None,
                "quant_format": "Q4_0",
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "acquire_gguf_artifact",
        lambda *args, **kwargs: _async_value(str(model_path)),
    )
    monkeypatch.setattr(
        module,
        "inspect_gguf_metadata",
        lambda path: {
            "model_family": "llama",
            "num_layers": 6,
            "hidden_size": 288,
            "num_attention_heads": 6,
            "num_key_value_heads": 6,
            "max_position_embeddings": 128,
            "vocab_size": 32000,
        },
    )
    orchestrator = Orchestrator(teardown_grace_seconds=0)
    monkeypatch.setattr(
        orchestrator,
        "_create_and_register_worker",
        lambda envelope, strategy: (
            orchestrator.register_worker(envelope.job_id, worker := Worker()) or worker
        ),
    )
    envelope = JobEnvelope(
        job_id=job_id,
        auth=AuthBlock(api_key="internal"),
        interface=InterfaceType.CLI,
        mode=JobMode.AUTO,
        model_source=ModelSource(repo_id="owner/model-gguf"),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )

    async def collect():
        return [event async for event in orchestrator.process_job(envelope)]

    import asyncio

    events = asyncio.run(collect())
    assert captured["model_path"] == str(model_path)
    assert captured["format"] == "Q4_0"
    assert events[-1].payload["status"] == "complete"


def test_orchestrator_blocks_non_validation_result():
    import asyncio

    job_id = uuid4()

    class InvalidWorker:
        async def validate(self, prompts):
            return {"outputs": ["looks plausible but is not validation"]}

    orchestrator = Orchestrator()
    orchestrator.register_worker(job_id, InvalidWorker())
    envelope = JobEnvelope(
        job_id=job_id,
        auth=AuthBlock(api_key="internal"),
        interface=InterfaceType.CLI,
        mode=JobMode.AUTO,
        model_source=ModelSource(repo_id="owner/model"),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )

    with pytest.raises(RuntimeError, match="artifact delivery is blocked"):
        asyncio.run(orchestrator._validate_registered_workers(envelope))


async def _async_value(value):
    return value
