import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.orchestrator import Orchestrator
from core.schemas import (
    ArtifactReference,
    AuthBlock,
    CallbackConfig,
    InterfaceType,
    JobEnvelope,
    JobMode,
    JobOperation,
    ModelSource,
    SystemPrompt,
    ValidationPolicy,
    ValidationPrompt,
)
from core.validation_service import (
    LlamaPerplexityEvaluator,
    validate_reference_candidate,
)


class FixedEvaluator:
    max_context_length = 256

    def __init__(self, value):
        self.value = value
        self.calls = []

    async def perplexity(self, text):
        self.calls.append(text)
        return self.value

    async def close(self):
        return None


def test_reference_candidate_validation_uses_absolute_deltas_and_weights():
    reference = FixedEvaluator(10.0)
    candidate = FixedEvaluator(10.5)

    result = asyncio.run(
        validate_reference_candidate(
            reference,
            candidate,
            policy=ValidationPolicy(),
        )
    )

    assert set(result.per_domain) == {"logic", "retrieval", "code"}
    assert result.composite_delta == pytest.approx(0.5)
    assert result.severity_tier == "poor"
    assert result.requires_confirmation is True
    assert len(reference.calls) == 11
    assert len(candidate.calls) == 11


def test_selected_domains_are_renormalized_and_golden_is_separate():
    result = asyncio.run(
        validate_reference_candidate(
            FixedEvaluator(5.0),
            FixedEvaluator(5.1),
            policy=ValidationPolicy(domains=["logic"]),
            golden_prompts=[
                ValidationPrompt(prompt="critical behavior", expected_output="yes")
            ],
        )
    )

    assert result.composite_delta == pytest.approx(0.1)
    assert list(result.per_domain) == ["logic"]
    assert len(result.golden_results) == 1
    assert result.golden_results[0].passed is True


def test_llama_perplexity_parser_accepts_realistic_final_line():
    output = "perplexity: calculating\\nFinal estimate: PPL = 12.3456"

    assert LlamaPerplexityEvaluator._PPL_PATTERN.findall(output)[-1] == "12.3456"


def test_validate_operation_persists_strict_result(tmp_path, monkeypatch):
    import core.orchestrator as module
    from telemetry.db import create_database, get_job, get_validation_result

    monkeypatch.setenv("HARADIBOTS_CACHE_ROOT", str(tmp_path / "cache"))
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
    evaluators = iter([FixedEvaluator(10.0), FixedEvaluator(10.1)])

    async def fake_build(source):
        return next(evaluators)

    monkeypatch.setattr(module, "build_evaluator", fake_build)
    candidate = tmp_path / "candidate.gguf"
    candidate.write_bytes(b"GGUF")
    envelope = JobEnvelope(
        job_id=uuid4(),
        auth=AuthBlock(api_key="internal"),
        interface=InterfaceType.CLI,
        mode=JobMode.AUTO,
        operation=JobOperation.VALIDATE,
        source_model=ModelSource(repo_id="owner/reference"),
        candidate_artifact=ArtifactReference(
            local_path=str(candidate),
            format="gguf",
        ),
        validation_policy=ValidationPolicy(),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )

    events = asyncio.run(
        _collect(Orchestrator(teardown_grace_seconds=0), envelope)
    )

    assert events[-1].payload["status"] == "complete"
    engine = create_database()
    try:
        assert get_job(engine, str(envelope.job_id)).state == "VALIDATED"
        stored = get_validation_result(engine, str(envelope.job_id))
        assert stored["severity"] == "good"
        assert set(stored["per_domain"]) == {"logic", "retrieval", "code"}
    finally:
        engine.dispose()


async def _collect(orchestrator, envelope):
    return [event async for event in orchestrator.process_job(envelope)]
