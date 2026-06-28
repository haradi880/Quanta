import pytest
from pydantic import ValidationError

from core.schemas import (
    ArtifactReference,
    AuthBlock,
    CallbackConfig,
    HardwareProfile,
    JobEnvelope,
    JobMode,
    JobOperation,
    InterfaceType,
    ModelSource,
    ModelMetaProfile,
    ProgressEvent,
    SystemPrompt,
    ValidationPolicy,
)


def test_wire_contract_field_names_are_exact():
    assert set(JobEnvelope.model_fields) == {
        "schema_version",
        "job_id",
        "auth",
        "interface",
        "mode",
        "operation",
        "source_model",
        "candidate_artifact",
        "target",
        "validation_policy",
        "hardware_override",
        "quantization_override",
        "cluster_config",
        "validation_prompts",
        "system_prompt",
        "telemetry_interval_ms",
        "callbacks",
    }
    assert set(ProgressEvent.model_fields) == {
        "schema_version",
        "job_id",
        "event_type",
        "timestamp_utc",
        "payload",
        "telemetry",
    }


def test_auth_is_required_and_contains_exactly_one_credential():
    assert JobEnvelope.model_fields["auth"].is_required()
    with pytest.raises(ValidationError):
        AuthBlock(api_key="key", jwt_token="token")
    with pytest.raises(ValidationError):
        AuthBlock()


def test_referenced_profile_fields_are_present():
    assert "gpu_uuids" in HardwareProfile.model_fields
    assert {
        "repo_exists",
        "is_gated",
        "repo_size_bytes",
        "file_manifest",
        "num_shards",
        "total_weight_bytes",
        "quant_bits",
        "attention_type",
        "kv_head_ratio",
    } <= set(ModelMetaProfile.model_fields)


def test_v31_operation_requirements_are_fail_closed():
    common = {
        "job_id": __import__("uuid").uuid4(),
        "auth": AuthBlock(api_key="test"),
        "interface": InterfaceType.CLI,
        "mode": JobMode.AUTO,
        "source_model": ModelSource(repo_id="owner/source"),
        "validation_policy": ValidationPolicy(),
        "system_prompt": SystemPrompt(preset_id="default"),
        "callbacks": CallbackConfig(),
    }
    with pytest.raises(ValidationError, match="quantize requires target"):
        JobEnvelope(operation=JobOperation.QUANTIZE, **common)
    with pytest.raises(ValidationError, match="validate requires candidate"):
        JobEnvelope(operation=JobOperation.VALIDATE, **common)
    valid = JobEnvelope(
        operation=JobOperation.VALIDATE,
        candidate_artifact=ArtifactReference(local_path="candidate.gguf"),
        **common,
    )
    assert valid.schema_version == "3.1"
