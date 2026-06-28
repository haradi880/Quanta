import asyncio
from uuid import uuid4

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
    SystemPrompt,
    ValidationPolicy,
)


def test_all_degradation_tiers_apply_architecture_actions():
    tier1 = Orchestrator._degrade(1, "no GPU", strategy={"backend": "vLLM"})
    assert tier1["strategy"]["backend"] == "llama.cpp"
    assert tier1["strategy"]["gpu_layers"] == 0

    tier2 = Orchestrator._degrade(2, "low VRAM", strategy={})
    assert tier2["strategy"]["partial_offload"] is True
    assert tier2["strategy"]["gpu_layers"] == "calculated"

    tier3 = Orchestrator._degrade(3, "OOM", strategy={"batch_size": 16})
    assert tier3["strategy"]["batch_size"] == 8
    assert tier3["strategy"]["max_retries"] == 3
    assert tier3["retry"] is True

    tier4 = Orchestrator._degrade(
        4,
        "GPU exhausted",
        strategy={"backend": "AutoAWQ"},
        hardware={
            "cpu": {
                "physical_cores": 12,
                "p_core_ids": [0, 2, 4, 6],
            }
        },
    )
    assert tier4["strategy"]["backend"] == "llama.cpp"
    assert tier4["strategy"]["threads"] == 4
    assert tier4["strategy"]["thread_affinity"] == [0, 2, 4, 6]

    tier5 = Orchestrator._degrade(5, "RAM exhausted", strategy={})
    assert tier5["abort"] is True
    assert tier5["preserve_partial_artifacts"] is True


def test_teardown_accepts_process_that_exits_before_signal():
    class ExitedProcess:
        pid = 987654
        returncode = 0

        def terminate(self):
            raise ProcessLookupError

    class Worker:
        process = ExitedProcess()

    job_id = uuid4()
    orchestrator = Orchestrator(teardown_grace_seconds=0)
    orchestrator.register_worker(job_id, Worker())

    event = asyncio.run(orchestrator._teardown(job_id))

    assert event.event_type is EventType.TEARDOWN_COMPLETE
    assert event.payload["harvested_pids"] == [987654]
    assert event.payload["forced_kill_count"] == 0
    assert orchestrator.worker_handles(job_id) == ()


def test_oom_retries_three_times_then_uses_cpu_fallback(tmp_path):
    orchestrator = Orchestrator(teardown_grace_seconds=0)
    calls = []
    model_path = tmp_path / "source"
    model_path.mkdir()

    async def fake_execute(envelope, strategy):
        calls.append(dict(strategy))
        if len(calls) <= 4:
            if False:
                yield None
            raise RuntimeError("CUDA out of memory")
        if False:
            yield None

    orchestrator._execute_registered_workers = fake_execute
    envelope = JobEnvelope(
        job_id=uuid4(),
        auth=AuthBlock(api_key="test"),
        interface=InterfaceType.CLI,
        mode=JobMode.AUTO,
        operation=JobOperation.QUANTIZE,
        source_model=ModelSource(repo_id="owner/model"),
        target={"format": "AWQ_INT4"},
        validation_policy=ValidationPolicy(),
        system_prompt=SystemPrompt(preset_id="default"),
        callbacks=CallbackConfig(),
    )
    strategy = {
        "backend": "AutoAWQ",
        "format": "AWQ_INT4",
        "batch_size": 16,
        "model_path": str(model_path),
    }
    hardware = {
        "cpu": {
            "physical_cores": 8,
            "p_core_ids": [0, 2, 4, 6],
        }
    }

    async def collect():
        return [
            event
            async for event in orchestrator._execute_with_degradation(
                envelope,
                strategy,
                hardware,
            )
        ]

    events = asyncio.run(collect())
    tiers = [event.payload["tier"] for event in events]
    assert tiers == [3, 3, 3, 4]
    assert calls[-1]["backend"] == "llama.cpp"
    assert calls[-1]["format"] == "Q4_K_M"
    assert calls[-1]["threads"] == 4
    assert all(event.event_type is EventType.DEGRADATION_WARNING for event in events)
