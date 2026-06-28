import asyncio

import aiohttp
import pytest

import core.hf_inspector as inspector


@pytest.mark.parametrize(
    ("tokenizer", "model_type", "expected"),
    [
        ({"chat_template": "<|start_header_id|>"}, "llama", "llama3"),
        ({"chat_template": "<|im_start|>"}, "qwen", "chatml"),
        ({"chat_template": "[INST] x"}, "mistral", "mistral"),
        ({"chat_template": "<|system|>"}, "phi3", "phi3"),
        ({}, "gemma", "gemma"),
        ({"chat_template": "other"}, "unknown", "custom"),
    ],
)
def test_chat_template_classification(tokenizer, model_type, expected):
    assert inspector._classify_chat_template(tokenizer, model_type) == expected


@pytest.mark.parametrize(
    ("config", "files", "expected"),
    [
        ({"quantization_config": {"quant_method": "AWQ"}}, [], (True, "awq")),
        ({}, ["model-gptq.safetensors"], (True, "gptq")),
        ({}, ["model-exl2.safetensors"], (True, "exl2")),
        ({}, ["model.Q4_K_M.gguf"], (True, "Q4_K_M")),
        ({}, ["model.gguf"], (True, "gguf")),
        ({}, ["model.safetensors"], (False, None)),
    ],
)
def test_quantization_detection(config, files, expected):
    assert inspector._detect_quantization(config, files) == expected


def test_metadata_helpers_and_headers(monkeypatch):
    assert inspector._first_int({"a": True, "b": -1, "c": 3}, "a", "b", "c") == 3
    assert inspector._quant_bits({"quantization_config": {"bits": 4}}, None) == 4
    assert inspector._quant_bits({}, "EXL2_3.5BPW") == 3.5
    assert inspector._estimate_parameter_count({}, None) is None
    assert inspector._estimate_parameter_count({"model.safetensors": 200}, None) == 100
    monkeypatch.setenv("HF_TOKEN", "secret")
    headers = inspector._request_headers()
    assert headers["Authorization"] == "Bearer secret"


def test_full_inspection_builds_gqa_profile(monkeypatch):
    metadata = {
        "sha": "revision",
        "gated": False,
        "usedStorage": 1000,
        "safetensors": {"total": 1_000_000},
        "siblings": [
            {"rfilename": "model-00001-of-00002.safetensors", "size": 500},
            {"rfilename": "model-00002-of-00002.safetensors", "size": 500},
            {"rfilename": "README.md", "size": 10},
        ],
    }
    config = {
        "model_type": "llama",
        "num_hidden_layers": 16,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
    }

    async def fake_metadata(repo_id, *, session):
        return metadata

    async def fake_file(repo_id, revision, filename, *, session):
        if filename == "config.json":
            return config
        return {"chat_template": "<|start_header_id|>"}

    monkeypatch.setattr(inspector, "_fetch_repo_metadata", fake_metadata)
    monkeypatch.setattr(inspector, "_fetch_json_file", fake_file)

    profile = asyncio.run(inspector.inspect_repo("owner/model"))

    assert profile["attention_type"] == "gqa"
    assert profile["kv_head_ratio"] == 0.25
    assert profile["num_shards"] == 2
    assert profile["total_weight_bytes"] == 1000
    assert profile["chat_template_type"] == "llama3"


def test_missing_repository_returns_conservative_profile(monkeypatch):
    async def missing(repo_id, *, session):
        return None

    monkeypatch.setattr(inspector, "_fetch_repo_metadata", missing)
    profile = asyncio.run(inspector.inspect_repo("owner/missing"))
    assert profile["repo_exists"] is False
    assert profile["upper_bound_only"] is True


class _Response:
    def __init__(self, status=200, payload=None, text="", content_length=None):
        self.status = status
        self.payload = payload
        self._text = text
        self.content_length = content_length

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self, **kwargs):
        return self.payload

    async def text(self):
        return self._text


class _Session:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, None), (200, {"sha": "abc"})],
)
def test_fetch_repo_metadata_statuses(status, expected):
    session = _Session(_Response(status=status, payload=expected))
    assert (
        asyncio.run(inspector._fetch_repo_metadata("owner/model", session=session))
        == expected
    )


def test_fetch_repo_metadata_rejects_bad_inputs_and_responses():
    with pytest.raises(ValueError, match="owner/name"):
        asyncio.run(inspector._fetch_repo_metadata("invalid", session=_Session()))
    with pytest.raises(inspector.HuggingFaceAccessError, match="gated"):
        asyncio.run(
            inspector._fetch_repo_metadata(
                "owner/model", session=_Session(_Response(status=403))
            )
        )
    with pytest.raises(inspector.HuggingFaceInspectionError, match="HTTP 500"):
        asyncio.run(
            inspector._fetch_repo_metadata(
                "owner/model",
                session=_Session(_Response(status=500, text="failure")),
            )
        )
    with pytest.raises(inspector.HuggingFaceInspectionError, match="malformed"):
        asyncio.run(
            inspector._fetch_repo_metadata(
                "owner/model",
                session=_Session(_Response(payload=["not", "object"])),
            )
        )
    with pytest.raises(inspector.HuggingFaceInspectionError, match="request failed"):
        asyncio.run(
            inspector._fetch_repo_metadata(
                "owner/model",
                session=_Session(error=aiohttp.ClientConnectionError()),
            )
        )


def test_fetch_json_file_status_size_and_payload_rules():
    assert (
        asyncio.run(
            inspector._fetch_json_file(
                "owner/model",
                "main",
                "config.json",
                session=_Session(_Response(status=404)),
            )
        )
        is None
    )
    with pytest.raises(inspector.HuggingFaceAccessError, match="authorized"):
        asyncio.run(
            inspector._fetch_json_file(
                "owner/model",
                "main",
                "config.json",
                session=_Session(_Response(status=401)),
            )
        )
    with pytest.raises(inspector.HuggingFaceInspectionError, match="HTTP 502"):
        asyncio.run(
            inspector._fetch_json_file(
                "owner/model",
                "main",
                "config.json",
                session=_Session(_Response(status=502)),
            )
        )
    with pytest.raises(inspector.HuggingFaceInspectionError, match="oversized"):
        asyncio.run(
            inspector._fetch_json_file(
                "owner/model",
                "main",
                "config.json",
                session=_Session(_Response(content_length=5_000_001)),
            )
        )
    assert (
        asyncio.run(
            inspector._fetch_json_file(
                "owner/model",
                "main",
                "config.json",
                session=_Session(_Response(payload=["invalid"])),
            )
        )
        is None
    )
