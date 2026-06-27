import pytest
from pydantic import ValidationError

from core.schemas import (
    AuthBlock,
    HardwareProfile,
    JobEnvelope,
    ModelMetaProfile,
    ProgressEvent,
)


def test_wire_contract_field_names_are_exact():
    assert set(JobEnvelope.model_fields) == {
        "schema_version",
        "job_id",
        "auth",
        "interface",
        "mode",
        "model_source",
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
