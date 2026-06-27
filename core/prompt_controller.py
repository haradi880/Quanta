"""Model-family-aware system prompt formatting."""

from __future__ import annotations


def format_system_prompt(raw_prompt: str, model_family: str | None) -> str:
    """Wrap a non-empty system prompt in the selected model's native tokens."""

    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("raw_prompt must be a non-empty string")
    family = (model_family or "unknown").lower().replace("_", "").replace("-", "")
    prompt = raw_prompt.strip()

    if family in {"llama3", "llama31", "llama32"}:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
        )
    if family.startswith("mistral") or family.startswith("mixtral"):
        return f"[INST] {prompt} [/INST]"
    if family in {"chatml", "qwen", "qwen2", "yi"}:
        return f"<|im_start|>system\n{prompt}<|im_end|>\n"
    if family.startswith("phi3"):
        return f"<|system|>\n{prompt}<|end|>\n"
    if family.startswith("gemma"):
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
    return raw_prompt
