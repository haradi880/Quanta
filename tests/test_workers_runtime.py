import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.schemas import EventType
from engines.awq_worker import AWQWorker
from engines.exl2_worker import EXL2Worker
from engines.gguf_worker import GGUFWorker
from engines.vllm_worker import VLLMWorker


async def collect(stream):
    return [event async for event in stream]


class AsyncLines:
    def __init__(self, lines):
        self.lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.lines:
            raise StopAsyncIteration
        return self.lines.pop(0)


class FakeProcess:
    def __init__(self, lines=(), return_code=0):
        self.stdout = AsyncLines(lines)
        self.returncode = None
        self.pid = 123
        self._return_code = return_code
        self.terminated = False

    async def wait(self):
        self.returncode = self._return_code
        return self.returncode

    async def communicate(self):
        self.returncode = self._return_code
        return b"", b""

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.returncode = -9


def test_awq_execute_validate_and_terminate(tmp_path, monkeypatch):
    class Model:
        def __init__(self):
            self.quant_config = None

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return cls()

        def quantize(self, tokenizer, quant_config):
            self.quant_config = quant_config

        def save_quantized(self, destination):
            (tmp_path / "output" / "weights.bin").write_bytes(b"weights")

        def generate(self, **encoded):
            return [[1, 2]]

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, source):
            return cls()

        def save_pretrained(self, destination):
            return None

        def __call__(self, text, return_tensors=None):
            return {"input_ids": [1]}

        def decode(self, tokens, skip_special_tokens=True):
            return "decoded"

    monkeypatch.setattr(
        AWQWorker,
        "_import_backend",
        classmethod(lambda cls: (Model, Tokenizer)),
    )
    worker = AWQWorker(uuid4())
    events = asyncio.run(
        collect(
            worker.execute(
                {
                    "model_source": "owner/model",
                    "output_path": str(tmp_path / "output"),
                }
            )
        )
    )
    assert [event.payload["status"] for event in events] == ["started", "complete"]
    assert events[-1].payload["output_path"] == str((tmp_path / "output").resolve())
    result = asyncio.run(worker.validate(["hello"]))
    assert result["outputs"] == ["decoded"]
    asyncio.run(worker.terminate())
    assert worker.is_alive() is False


def test_awq_errors_are_structured(monkeypatch):
    monkeypatch.setattr(
        AWQWorker,
        "_import_backend",
        classmethod(lambda cls: (_ for _ in ()).throw(ImportError("missing"))),
    )
    worker = AWQWorker(uuid4())
    events = asyncio.run(collect(worker.execute({})))
    assert events[0].event_type is EventType.ERROR
    assert "ImportError" in events[0].payload["message"]
    with pytest.raises(RuntimeError, match="not loaded"):
        asyncio.run(worker.validate([]))


def test_vllm_load_validate_and_input_guards(monkeypatch):
    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text=f"out:{prompt}")])
                for prompt in prompts
            ]

    class Sampling:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        VLLMWorker,
        "_import_backend",
        classmethod(lambda cls: (LLM, Sampling)),
    )
    worker = VLLMWorker(uuid4())
    events = asyncio.run(
        collect(
            worker.execute(
                {
                    "model_source": "owner/model",
                    "tp_degree": 2,
                    "engine_kwargs": {"dtype": "float16"},
                }
            )
        )
    )
    assert events[-1].payload["tensor_parallel_size"] == 2
    result = asyncio.run(worker.validate(["a", {"prompt": "b"}]))
    assert result["outputs"] == ["out:a", "out:b"]
    asyncio.run(worker.terminate())
    assert worker.is_alive() is False

    worker = VLLMWorker(uuid4())
    for invalid in (
        {},
        {"model_source": "x", "tp_degree": 0},
        {"model_source": "x", "engine_kwargs": "bad"},
    ):
        events = asyncio.run(collect(worker.execute(invalid)))
        assert events[-1].event_type is EventType.ERROR


def test_exl2_command_runtime_validate_and_guards(tmp_path, monkeypatch):
    script = tmp_path / "convert.py"
    script.write_text("# converter", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()

    class Config:
        model_dir = None

        def prepare(self):
            return None

    class Model:
        def __init__(self, config):
            self.config = config

        def load_autosplit(self, cache):
            return None

    class Cache:
        def __init__(self, model, lazy):
            self.model = model

    class Tokenizer:
        def __init__(self, config):
            self.config = config

    class Generator:
        def __init__(self, model, cache, tokenizer):
            pass

        def warmup(self):
            return None

        def generate_simple(self, prompt, max_new_tokens):
            return f"generated:{prompt}"

    monkeypatch.setattr(
        EXL2Worker,
        "_import_backend",
        classmethod(
            lambda cls: (Model, Cache, Config, Tokenizer, Generator)
        ),
    )
    worker = EXL2Worker(uuid4())
    worker._output_path = str(tmp_path / "output")
    worker._load_runtime()
    result = asyncio.run(worker.validate(["hello", {"prompt": "world"}]))
    assert result["outputs"] == ["generated:hello", "generated:world"]

    command = worker._conversion_command(
        {
            "convert_script": str(script),
            "model_path": str(source),
            "work_path": str(tmp_path / "work"),
            "output_path": str(tmp_path / "output"),
            "bits": 4.5,
            "no_resume": True,
        }
    )
    assert command[-1] == "-nr"
    with pytest.raises(ValueError, match="between 2 and 8"):
        worker._conversion_command(
            {
                "convert_script": str(script),
                "model_path": str(source),
                "work_path": "work",
                "output_path": "output",
                "bits": 9,
            }
        )
    asyncio.run(worker.terminate())
    assert worker.is_alive() is False


def test_gguf_inference_quantization_validation_and_terminate(tmp_path, monkeypatch):
    binary = tmp_path / "llama-cli"
    quantizer = tmp_path / "llama-quantize"
    converter = tmp_path / "convert.py"
    model = tmp_path / "model.gguf"
    source = tmp_path / "source"
    for path in (binary, quantizer):
        path.write_bytes(b"binary")
    converter.write_text("# converter", encoding="utf-8")
    model.write_bytes(b"GGUF")
    source.mkdir()
    processes = []

    async def fake_subprocess(*command, **kwargs):
        process = FakeProcess([b"42% loading\n", b"answer\n"])
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    worker = GGUFWorker(uuid4(), binary_path=str(binary))
    events = asyncio.run(
        collect(
            worker.execute(
                {
                    "operation": "infer",
                    "model_path": str(model),
                    "prompt": "hello",
                    "max_tokens": 4,
                    "gpu_layers": 0,
                }
            )
        )
    )
    assert events[0].event_type is EventType.INFERENCE_PROGRESS
    assert any(event.payload.get("progress_pct") == 42 for event in events)
    result = asyncio.run(worker.validate([{"prompt": "check"}]))
    assert result["prompt_count"] == 1
    assert any("answer" in line for line in result["outputs"][0])

    output = tmp_path / "output.gguf"
    output.write_bytes(b"GGUF result")
    events = asyncio.run(
        collect(
            worker.execute(
                {
                    "operation": "quantize",
                    "model_path": str(source),
                    "work_path": str(tmp_path / "work"),
                    "output_path": str(output),
                    "format": "Q4_K_M",
                    "convert_script": str(converter),
                    "quantize_binary": str(quantizer),
                }
            )
        )
    )
    assert events[-1].payload["output_path"] == str(output.resolve())
    worker.process = FakeProcess()
    asyncio.run(worker.terminate())
    assert worker.is_alive() is False


def test_exl2_execute_streams_subprocess_progress(tmp_path, monkeypatch):
    script = tmp_path / "convert.py"
    script.write_text("# converter", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()

    class Placeholder:
        pass

    monkeypatch.setattr(
        EXL2Worker,
        "_import_backend",
        classmethod(lambda cls: (Placeholder,) * 5),
    )

    async def fake_subprocess(*command, **kwargs):
        return FakeProcess([b"calibrating\n"])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    worker = EXL2Worker(uuid4())
    events = asyncio.run(
        collect(
            worker.execute(
                {
                    "convert_script": str(script),
                    "model_path": str(source),
                    "work_path": str(tmp_path / "work"),
                    "output_path": str(tmp_path / "output"),
                    "bits": 4,
                }
            )
        )
    )
    assert [event.payload["status"] for event in events] == [
        "launched",
        "running",
        "complete",
    ]
