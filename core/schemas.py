"""Version 3.1 contracts shared across HaradiBots tiers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "3.1"


class StrictModel(BaseModel):
    """Base for wire contracts: strict types and no undeclared fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class InterfaceType(StrEnum):
    CLI = "cli"
    GUI = "gui"
    KAGGLE = "kaggle"
    API = "api"


class JobMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class JobOperation(StrEnum):
    INSPECT = "inspect"
    QUANTIZE = "quantize"
    VALIDATE = "validate"
    INFER = "infer"


class ClusterBackend(StrEnum):
    RAY = "ray"
    SLURM = "slurm"
    K8S = "k8s"


class EventType(StrEnum):
    HARDWARE_PROFILE = "hardware_profile"
    MODEL_INSPECTION = "model_inspection"
    STRATEGY_SELECTED = "strategy_selected"
    QUANTIZATION_PROGRESS = "quantization_progress"
    INFERENCE_PROGRESS = "inference_progress"
    VALIDATION_RESULT = "validation_result"
    TELEMETRY_TICK = "telemetry_tick"
    CLUSTER_NODE_STATUS = "cluster_node_status"
    CLUSTER_DEGRADED_WARNING = "cluster_degraded_warning"
    DEGRADATION_WARNING = "degradation_warning"
    EMERGENCY_STOP = "emergency_stop"
    TEARDOWN_COMPLETE = "teardown_complete"
    ERROR = "error"
    COMPLETE = "complete"


class AuthBlock(StrictModel):
    """Exactly one caller credential, as required by Architecture §1.2."""

    api_key: str | None = None
    jwt_token: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_credential(self) -> AuthBlock:
        if (self.api_key is None) == (self.jwt_token is None):
            raise ValueError("auth must contain either api_key or jwt_token, never both")
        return self


class ModelSource(StrictModel):
    """A Hugging Face repository or local model path, with an optional revision."""

    repo_id: str | None = None
    local_path: str | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_location(self) -> ModelSource:
        if (self.repo_id is None) == (self.local_path is None):
            raise ValueError("model reference requires exactly one of repo_id or local_path")
        return self


class ArtifactReference(ModelSource):
    format: str | None = None


class TargetConfig(StrictModel):
    format: str
    output_path: str | None = None
    quantization: dict[str, Any] = Field(default_factory=dict)


class ValidationPolicy(StrictModel):
    domains: list[Literal["logic", "retrieval", "code"]] = Field(
        default_factory=lambda: ["logic", "retrieval", "code"]
    )
    fail_closed: bool = True
    require_golden: bool = False

    @model_validator(mode="after")
    def require_unique_domains(self) -> ValidationPolicy:
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("validation domains must be non-empty and unique")
        return self


class HardwareOverride(StrictModel):
    vram_bytes: int | None = Field(default=None, ge=0)
    cpu_core_count: int | None = Field(default=None, ge=1)
    system_ram_bytes: int | None = Field(default=None, ge=0)


class QuantizationOverride(StrictModel):
    target_format: str
    gpu_layers: int | None = Field(default=None, ge=0)


class ClusterConfig(StrictModel):
    backend: ClusterBackend
    node_count: int = Field(ge=1)
    gpus_per_node: int = Field(ge=0)


class ValidationPrompt(StrictModel):
    prompt: str
    expected_output: str | None = None


class SystemPrompt(StrictModel):
    preset_id: str | None = None
    custom_text: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_prompt_source(self) -> SystemPrompt:
        if (self.preset_id is None) == (self.custom_text is None):
            raise ValueError("system_prompt requires either preset_id or custom_text")
        return self


class CallbackConfig(StrictModel):
    progress_channel: str | None = None
    completion_channel: str | None = None


class JobEnvelope(StrictModel):
    """Inbound Interface-to-Orchestrator contract from Architecture §1.2."""

    schema_version: Literal["3.1"] = SCHEMA_VERSION
    job_id: UUID
    auth: AuthBlock
    interface: InterfaceType
    mode: JobMode
    operation: JobOperation
    source_model: ModelSource
    candidate_artifact: ArtifactReference | None = None
    target: TargetConfig | None = None
    validation_policy: ValidationPolicy
    hardware_override: HardwareOverride | None = None
    quantization_override: QuantizationOverride | None = None
    cluster_config: ClusterConfig | None = None
    validation_prompts: list[ValidationPrompt] | None = None
    system_prompt: SystemPrompt
    telemetry_interval_ms: int = Field(default=1000, ge=1)
    callbacks: CallbackConfig

    @model_validator(mode="after")
    def enforce_operation_requirements(self) -> JobEnvelope:
        if self.operation is JobOperation.QUANTIZE and self.target is None:
            raise ValueError("quantize requires target")
        if self.operation is JobOperation.VALIDATE and self.candidate_artifact is None:
            raise ValueError("validate requires candidate_artifact")
        if self.validation_policy.require_golden and not self.validation_prompts:
            raise ValueError("validation policy requires at least one golden prompt")
        return self


class ProgressEvent(StrictModel):
    """Outbound Orchestrator-to-Interface stream contract from §1.2."""

    schema_version: Literal["3.1"] = SCHEMA_VERSION
    job_id: UUID
    event_type: EventType
    timestamp_utc: datetime
    payload: dict[str, Any]
    telemetry: dict[str, Any]


class GPUProfile(StrictModel):
    uuid: str
    vram_total_bytes: int = Field(ge=0)
    vram_free_bytes: int = Field(ge=0)
    cuda_cc_major: int | None = Field(default=None, ge=0)
    cuda_cc_minor: int | None = Field(default=None, ge=0)
    mem_bandwidth_gb_s: float | None = Field(default=None, ge=0)
    gpu_temp_c: float | None = None
    power_draw_w: float | None = Field(default=None, ge=0)
    power_limit_w: float | None = Field(default=None, ge=0)
    nvlink_peers: list[str] = Field(default_factory=list)


class CPUProfile(StrictModel):
    ram_total_gb: float = Field(ge=0)
    ram_available_gb: float = Field(ge=0)
    physical_cores: int = Field(ge=1)
    p_core_ids: list[int] = Field(default_factory=list)
    e_core_ids: list[int] = Field(default_factory=list)
    core_topology: Literal["hybrid", "uniform", "unknown"]
    p_core_clock_ghz: float | None = Field(default=None, ge=0)
    e_core_clock_ghz: float | None = Field(default=None, ge=0)
    isa_flags: list[str] = Field(default_factory=list)
    degraded_topology_detection: bool = False


class HardwareProfile(StrictModel):
    profile_id: UUID
    timestamp_utc: datetime
    gpu_count: int = Field(ge=0)
    gpu_uuids: list[str] = Field(default_factory=list)
    gpus: list[GPUProfile] = Field(default_factory=list)
    cpu: CPUProfile

    @model_validator(mode="after")
    def match_gpu_inventory(self) -> HardwareProfile:
        if self.gpu_count != len(self.gpus):
            raise ValueError("gpu_count must match the number of GPU profiles")
        if self.gpu_uuids != [gpu.uuid for gpu in self.gpus]:
            raise ValueError("gpu_uuids must match GPU profile order")
        return self


class StrategyConfig(StrictModel):
    format: str
    gpu_layers: int = Field(ge=0)
    backend: str
    tp_degree: int = Field(default=1, ge=1)
    pp_degree: int = Field(default=1, ge=1)
    dp_degree: int = Field(default=1, ge=1)
    warning: bool = False
    warning_reason: str | None = None


class ErrorEnvelope(StrictModel):
    schema_version: Literal["3.1"] = SCHEMA_VERSION
    job_id: UUID | None = None
    code: int
    error: str
    message: str
    timestamp_utc: datetime


class TeardownComplete(StrictModel):
    schema_version: Literal["3.1"] = SCHEMA_VERSION
    job_id: UUID
    harvested_pids: list[int]
    forced_kill_count: int = Field(ge=0)
    timestamp_utc: datetime


class ModelMetaProfile(StrictModel):
    repo_id: str
    repo_exists: bool
    is_gated: bool
    repo_size_bytes: int = Field(ge=0)
    parameter_count: int | None = Field(default=None, ge=0)
    file_manifest: dict[str, int]
    num_shards: int = Field(ge=0)
    total_weight_bytes: int = Field(ge=0)
    num_layers: int | None = Field(default=None, ge=0)
    hidden_size: int | None = Field(default=None, ge=0)
    num_attention_heads: int | None = Field(default=None, ge=0)
    num_key_value_heads: int | None = Field(default=None, ge=0)
    vocab_size: int | None = Field(default=None, ge=0)
    max_position_embeddings: int | None = Field(default=None, ge=0)
    attention_type: Literal["gqa", "mha", "mqa"]
    kv_head_ratio: float | None = Field(default=None, ge=0)
    upper_bound_only: bool = False
    chat_template_type: str | None = None
    is_prequantized: bool = False
    quant_format: str | None = None
    quant_bits: float | None = Field(default=None, gt=0)
    model_family: str | None = None


class DomainValidationResult(StrictModel):
    original_perplexity: float = Field(ge=0)
    quantized_perplexity: float = Field(ge=0)
    delta: float


class GoldenValidationResult(StrictModel):
    prompt: str
    expected_output: str | None = None
    actual_output: str | None = None
    passed: bool | None = None


class ValidationResult(StrictModel):
    per_domain: dict[str, DomainValidationResult]
    composite_delta: float
    severity_tier: Literal["excellent", "good", "moderate", "poor", "critical"]
    requires_confirmation: bool = False
    quarantined: bool = False
    golden_results: list[GoldenValidationResult] = Field(default_factory=list)
