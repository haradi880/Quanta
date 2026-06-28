import asyncio
import struct
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from core.artifacts import (
    ArtifactCompatibilityError,
    acquire_gguf_artifact,
    acquire_source_snapshot,
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
    JobOperation,
    ModelSource,
    ProgressEvent,
    SystemPrompt,
    TargetConfig,
    ValidationPolicy,
    ValidationResult,
)
from engines.gguf_worker import GGUFWorker


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
        + struct.pack("<Q", 1)
        + struct.pack("<Q", len(entries))
        + b"".join(entries)
        + _gguf_string("output.weight")
        + struct.pack("<I", 2)
        + struct.pack("<Q", 3)
        + struct.pack("<Q", 2)
        + struct.pack("<I", 0)
        + struct.pack("<Q", 0)
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
        "parameter_count": 6,
    }


def test_local_gguf_profile_is_plannable_without_hugging_face(tmp_path):
    model = tmp_path / "tiny-Q4_0.gguf"
    entries = [
        _metadata_entry("general.architecture", 8, "llama"),
        _metadata_entry("llama.block_count", 4, 1),
        _metadata_entry("llama.embedding_length", 4, 4),
        _metadata_entry("llama.attention.head_count", 4, 2),
        _metadata_entry("llama.attention.head_count_kv", 4, 1),
        _metadata_entry("llama.context_length", 4, 128),
        _metadata_entry("tokenizer.ggml.tokens", 9, ["a", "b"]),
    ]
    model.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", len(entries))
        + b"".join(entries)
        + _gguf_string("weight")
        + struct.pack("<I", 2)
        + struct.pack("<Q", 4)
        + struct.pack("<Q", 4)
        + struct.pack("<I", 2)
        + struct.pack("<Q", 0)
    )

    profile = Orchestrator._inspect_local_gguf(model)

    assert profile["parameter_count"] == 16
    assert profile["quant_format"] == "Q4_0"
    assert profile["quant_bits"] == 4.0
    assert profile["attention_type"] == "mqa"
    assert profile["file_manifest"] == {model.name: model.stat().st_size}


def test_weight_size_parameter_fallback_is_conservative():
    assert _estimate_parameter_count({"pytorch_model.bin": 4000}, None) == 1000
    assert _estimate_parameter_count({"model-Q4_0.gguf": 500}, 4) == 1000


def test_sandboxed_gguf_and_source_acquisition(tmp_path, monkeypatch):
    import core.artifacts as artifacts

    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))

    def fake_download(**kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / kwargs["filename"]
        path.write_bytes(b"GGUF")
        return str(path)

    def fake_snapshot(**kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    monkeypatch.setattr(artifacts, "hf_hub_download", fake_download)
    monkeypatch.setattr(artifacts, "snapshot_download", fake_snapshot)
    metadata = {"file_manifest": {"model-Q4_K_M.gguf": 4}}
    path = asyncio.run(
        acquire_gguf_artifact("owner/model", metadata, "Q4_K_M")
    )
    assert Path(path).read_bytes() == b"GGUF"
    source = asyncio.run(acquire_source_snapshot("owner/source"))
    assert (Path(source) / "model.safetensors").is_file()


def test_source_snapshot_rejects_incomplete_download(tmp_path, monkeypatch):
    import core.artifacts as artifacts

    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))

    def missing_config(**kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        return str(destination)

    monkeypatch.setattr(artifacts, "snapshot_download", missing_config)
    with pytest.raises(OSError, match="config.json"):
        asyncio.run(acquire_source_snapshot("owner/source"))


def test_source_snapshot_rejects_missing_weights(tmp_path, monkeypatch):
    import core.artifacts as artifacts

    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))

    def config_only(**kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        return str(destination)

    monkeypatch.setattr(artifacts, "snapshot_download", config_only)
    with pytest.raises(OSError, match="no supported full-precision weights"):
        asyncio.run(acquire_source_snapshot("owner/source"))


def test_gguf_acquisition_rejects_storage_and_empty_download(tmp_path, monkeypatch):
    import core.artifacts as artifacts

    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        artifacts.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 1})(),
    )
    metadata = {"file_manifest": {"model.gguf": 100}}
    with pytest.raises(OSError, match="insufficient storage"):
        asyncio.run(acquire_gguf_artifact("owner/model", metadata, "Q4"))

    monkeypatch.setattr(
        artifacts.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 1000})(),
    )

    def empty_download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(artifacts, "hf_hub_download", empty_download)
    with pytest.raises(OSError, match="missing or empty"):
        asyncio.run(acquire_gguf_artifact("owner/model", metadata, "Q4"))


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"NOPE", "file is not GGUF"),
        (b"GGUF" + struct.pack("<I", 1), "unsupported GGUF version"),
        (
            b"GGUF"
            + struct.pack("<I", 3)
            + struct.pack("<Q", 0)
            + struct.pack("<Q", 100_001),
            "entry count exceeds",
        ),
    ],
)
def test_gguf_metadata_rejects_invalid_headers(tmp_path, payload, message):
    path = tmp_path / "bad.gguf"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        inspect_gguf_metadata(path)


def test_gguf_quantization_commands_support_conversion_and_requantization(
    tmp_path,
):
    converter = tmp_path / "convert.py"
    quantizer = tmp_path / "llama-quantize"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_bytes(b"binary")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output = tmp_path / "output" / "model.gguf"
    worker = GGUFWorker(uuid4())
    common = {
        "work_path": str(tmp_path / "work"),
        "output_path": str(output),
        "format": "Q4_K_M",
        "convert_script": str(converter),
        "quantize_binary": str(quantizer),
    }

    convert, quantize, resolved = worker._quantization_commands(
        {**common, "model_path": str(source_dir)}
    )
    assert convert is not None
    assert "--outtype" in convert
    assert quantize[-1] == "Q4_K_M"
    assert resolved == str(output.resolve())

    existing = tmp_path / "existing.gguf"
    existing.write_bytes(b"GGUF")
    convert, quantize, _ = worker._quantization_commands(
        {**common, "model_path": str(existing)}
    )
    assert convert is None
    assert quantize[0] == str(quantizer.resolve())
    assert quantize[1] == str(existing.resolve())


def test_gguf_frozen_conversion_uses_private_entrypoint(tmp_path, monkeypatch):
    import engines.gguf_worker as module

    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    quantizer = tmp_path / "llama-quantize.exe"
    quantizer.write_bytes(b"binary")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(module.sys, "frozen", True, raising=False)

    convert, _, _ = GGUFWorker(uuid4())._quantization_commands(
        {
            "model_path": str(source),
            "output_path": str(tmp_path / "output.gguf"),
            "work_path": str(tmp_path / "work"),
            "format": "Q4_K_M",
            "convert_script": str(converter),
            "quantize_binary": str(quantizer),
        }
    )

    assert convert is not None
    assert convert[1:3] == ["_convert-hf-to-gguf", str(converter.resolve())]


def test_quantization_target_controls_backend_not_hardware_recommendation():
    strategy = {"format": "AWQ_INT4", "backend": "AutoAWQ"}

    Orchestrator._apply_target_backend(strategy, "Q4_K_M", gpu_count=2)
    assert strategy == {"format": "Q4_K_M", "backend": "llama.cpp CUDA"}

    Orchestrator._apply_target_backend(strategy, "EXL2_4.0BPW", gpu_count=2)
    assert strategy["backend"] == "ExLlamaV2"

    with pytest.raises(RuntimeError, match="no production quantization backend"):
        Orchestrator._apply_target_backend(strategy, "GPTQ_INT4", gpu_count=1)


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
                schema_version="3.1",
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
        operation=JobOperation.INFER,
        source_model=ModelSource(repo_id="owner/model-gguf"),
        validation_policy=ValidationPolicy(),
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
        operation=JobOperation.INFER,
        source_model=ModelSource(repo_id="owner/model"),
        validation_policy=ValidationPolicy(),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )

    with pytest.raises(RuntimeError, match="artifact delivery is blocked"):
        asyncio.run(orchestrator._validate_registered_workers(envelope))


def test_quantization_output_is_handed_to_strict_validator(tmp_path, monkeypatch):
    import asyncio
    import core.orchestrator as module

    job_id = uuid4()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = tmp_path / "candidate.Q4_K_M.gguf"
    captured = {}

    class Worker:
        process = None

        async def execute(self, strategy):
            artifact.write_bytes(b"GGUF artifact")
            yield ProgressEvent(
                schema_version="3.1",
                job_id=job_id,
                event_type=EventType.QUANTIZATION_PROGRESS,
                timestamp_utc=datetime.now(timezone.utc),
                payload={"status": "complete", "output_path": str(artifact)},
                telemetry={},
            )

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
                "file_manifest": {"model.safetensors": 30_000_000},
                "num_layers": 6,
                "num_attention_heads": 6,
                "num_key_value_heads": 6,
                "hidden_size": 288,
                "max_position_embeddings": 128,
                "is_prequantized": False,
                "quant_format": None,
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "acquire_source_snapshot",
        lambda *args, **kwargs: _async_value(str(source_dir)),
    )
    orchestrator = Orchestrator(teardown_grace_seconds=0)
    monkeypatch.setattr(
        orchestrator,
        "_create_and_register_worker",
        lambda envelope, strategy: (
            orchestrator.register_worker(envelope.job_id, worker := Worker()) or worker
        ),
    )

    async def fake_validation(validation_envelope):
        captured["candidate"] = validation_envelope.candidate_artifact.local_path
        domain = DomainValidationResult(
            original_perplexity=10,
            quantized_perplexity=10.01,
            delta=0.01,
        )
        return ValidationResult(
            per_domain={"logic": domain, "retrieval": domain, "code": domain},
            composite_delta=0.01,
            severity_tier="excellent",
        )

    from core.schemas import DomainValidationResult

    monkeypatch.setattr(orchestrator, "_run_validation_operation", fake_validation)
    envelope = JobEnvelope(
        job_id=job_id,
        auth=AuthBlock(api_key="internal"),
        interface=InterfaceType.CLI,
        mode=JobMode.AUTO,
        operation=JobOperation.QUANTIZE,
        source_model=ModelSource(repo_id="owner/reference"),
        target=TargetConfig(format="Q4_K_M"),
        validation_policy=ValidationPolicy(),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )

    events = asyncio.run(_collect_events(orchestrator, envelope))
    assert captured["candidate"] == str(artifact)
    assert events[-1].payload["status"] == "complete"


async def _collect_events(orchestrator, envelope):
    return [event async for event in orchestrator.process_job(envelope)]


async def _async_value(value):
    return value
