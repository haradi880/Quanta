import asyncio
import json

from aiohttp import web
import pytest

from core.accelerator import calc_available_ctx, count_tokens_native
from core.prompt_controller import format_system_prompt


def test_llama3_system_prompt_tokens_are_exact():
    result = format_system_prompt("You are a helper", "llama3")

    assert result == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helper<|eot_id|>"
    )


def test_all_prompt_families_and_unknown_fallback():
    assert format_system_prompt("p", "mistral") == "[INST] p [/INST]"
    assert "<|im_start|>system" in format_system_prompt("p", "chatml")
    assert format_system_prompt("p", "phi-3").startswith("<|system|>")
    assert format_system_prompt("p", "gemma").startswith("<start_of_turn>user")
    assert format_system_prompt("  raw  ", "unknown") == "  raw  "
    with pytest.raises(ValueError, match="non-empty"):
        format_system_prompt("   ", "llama3")


def test_persona_file_contains_three_non_deletable_presets():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config" / "persona_presets.json"
    presets = json.loads(path.read_text(encoding="utf-8"))["presets"]

    assert len(presets) == 3
    assert {preset["id"] for preset in presets} == {
        "json-api-extractor",
        "complexity-code-reviewer",
        "formal-logic-verifier",
    }
    assert all(preset["deletable"] is False for preset in presets)


def test_native_token_count_matches_backend():
    async def scenario():
        async def tokenize(request):
            return web.json_response({"count": 7})

        app = web.Application()
        app.router.add_post("/tokenize", tokenize)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await count_tokens_native("exact text", f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    assert asyncio.run(scenario()) == 7


def test_offline_token_count_adds_five_percent():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(100))

    assert asyncio.run(count_tokens_native("text", None, Tokenizer())) == 105


def test_context_budget_and_overflow_breakdown():
    assert calc_available_ctx(4096, 200, 100, online=True) == 3796
    assert calc_available_ctx(4096, 200, 100, online=False) == 3781
    overflow = calc_available_ctx(512, 200, 100, online=False)
    assert overflow["error"] == "context_overflow_error"
    assert overflow["safety_reserve"] == 15
