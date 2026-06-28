import asyncio
from types import SimpleNamespace

import pytest

import core.accelerator as accelerator


def model_meta():
    return {
        "parameter_count": 7_000_000_000,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 4096,
    }


def test_accelerator_format_parsing_and_input_guards():
    assert accelerator._bit_width("FP32") == 32
    assert accelerator._bit_width("BF16") == 16
    assert accelerator._bit_width("AWQ_INT4") == 4
    assert accelerator._bit_width("GPTQ") == 4
    assert accelerator._bit_width("Q6_K") == 6
    assert accelerator._bit_width("EXL2") == 4
    with pytest.raises(ValueError, match="cannot determine"):
        accelerator._bit_width("unknown")
    with pytest.raises(TypeError):
        accelerator._as_dict("bad")
    with pytest.raises(ValueError, match="parameter count"):
        accelerator._parameter_count({})
    with pytest.raises(ValueError, match="layer count"):
        accelerator._total_layers({})


def test_auto_and_manual_strategy_with_vram_warning():
    hardware = {
        "gpu_count": 1,
        "gpus": [{"vram_free_bytes": 4 * 1024**3}],
    }
    automatic = accelerator.select_strategy(hardware, model_meta())
    assert automatic["backend"] == "llama.cpp CUDA"
    manual = accelerator.select_strategy(
        hardware,
        model_meta(),
        mode="manual",
        override={
            "target_format": "FP16",
            "gpu_layers": 32,
            "backend": "vLLM",
        },
    )
    assert manual["warning"] is True
    assert "VRAM" in manual["warning_reason"]
    with pytest.raises(ValueError, match="mode"):
        accelerator.select_strategy(hardware, model_meta(), mode="invalid")


@pytest.mark.parametrize(
    ("source", "target", "allowed"),
    [
        ("AWQ", "Q4_K_M", False),
        ("GPTQ", "Q4_K_M", False),
        ("EXL2", "Q4_K_M", False),
        ("Q4_K_M", "Q3_K_M", False),
        ("FP16", "Q4_K_M", True),
        ("BF16", "Q8_0", True),
        ("Q8_0", "Q6_K", True),
        ("Q6_K", "Q4_K_M", True),
        ("Q5_K", "Q4_K_M", False),
    ],
)
def test_complete_conversion_safety_table(source, target, allowed):
    assert accelerator.check_overcompilation(source, target)["allowed"] is allowed


def test_perplexity_logits_and_output_guards():
    torch = pytest.importorskip("torch")

    class LogitsModel:
        def __call__(self, input_ids, labels):
            logits = torch.zeros((1, input_ids.shape[1], 10))
            return SimpleNamespace(loss=None, logits=logits)

    assert accelerator.calc_perplexity(LogitsModel(), [1, 2, 3]) == pytest.approx(10)
    with pytest.raises(ValueError, match="at least two"):
        accelerator.calc_perplexity(LogitsModel(), [1])

    class Empty:
        def log_probabilities(self, tokens):
            return []

    with pytest.raises(ValueError, match="no log probabilities"):
        accelerator.calc_perplexity(Empty(), [1, 2])


def test_token_count_response_offline_and_context_guards():
    assert accelerator._token_count_from_response({"count": 3}) == 3
    assert accelerator._token_count_from_response({"tokens": [1, 2]}) == 2
    with pytest.raises(ValueError, match="no token count"):
        accelerator._token_count_from_response({})

    class CallableTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(10))}

    assert accelerator._offline_token_count("x", CallableTokenizer()) == 11
    with pytest.raises(RuntimeError, match="requires"):
        accelerator._offline_token_count("x", None)
    with pytest.raises(ValueError, match="positive"):
        accelerator.calc_available_ctx(0, 0, 0, True)
    with pytest.raises(ValueError, match="negative"):
        accelerator.calc_available_ctx(100, -1, 0, True)


def test_native_token_count_falls_back_after_http_failure():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(20))

    result = asyncio.run(
        accelerator.count_tokens_native(
            "text",
            "http://127.0.0.1:1",
            Tokenizer(),
        )
    )
    assert result == 21
