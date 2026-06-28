"""Reference-versus-candidate validation with real backend evaluators."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol

from core.accelerator import calc_perplexity, generate_retrieval_prompt, get_severity_tier
from core.schemas import (
    ArtifactReference,
    DomainValidationResult,
    GoldenValidationResult,
    ModelSource,
    ValidationPolicy,
    ValidationPrompt,
    ValidationResult,
)


class PerplexityEvaluator(Protocol):
    max_context_length: int

    async def perplexity(self, text: str) -> float: ...

    async def close(self) -> None: ...


class TransformersEvaluator:
    """Lazy Transformers evaluator for canonical or candidate repositories."""

    def __init__(self, model_source: str, revision: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers validation dependencies are unavailable") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            revision=revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            revision=revision,
        )
        self.model.eval()
        configured = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        valid = [
            value
            for value in (configured, tokenizer_limit)
            if isinstance(value, int) and 128 <= value < 1_000_000
        ]
        self.max_context_length = min(valid) if valid else 2048

    async def perplexity(self, text: str) -> float:
        def calculate() -> float:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length,
            )
            return calc_perplexity(self.model, encoded["input_ids"])

        return await asyncio.to_thread(calculate)

    async def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class LlamaPerplexityEvaluator:
    """Subprocess-isolated GGUF evaluator using llama-perplexity."""

    _PPL_PATTERN = re.compile(
        r"(?:final estimate:?\s*)?(?:ppl|perplexity)\s*=\s*"
        r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model_path: str,
        *,
        binary_path: str | None = None,
        max_context_length: int = 2048,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.binary_path = binary_path or os.environ.get(
            "HARADIBOTS_LLAMA_PERPLEXITY_BIN"
        )
        if not self.binary_path or not Path(self.binary_path).is_file():
            raise RuntimeError(
                "GGUF validation requires HARADIBOTS_LLAMA_PERPLEXITY_BIN"
            )
        self.max_context_length = max(max_context_length, 128)

    async def perplexity(self, text: str) -> float:
        cache_root = Path(
            os.environ.get(
                "HARADIBOTS_CACHE_ROOT",
                str(Path.home() / ".haradibots" / "cache"),
            )
        ).expanduser()
        work_dir = cache_root / "validation"
        work_dir.mkdir(parents=True, exist_ok=True)
        descriptor, text_path = tempfile.mkstemp(
            prefix="ppl-",
            suffix=".txt",
            dir=work_dir,
            text=True,
        )
        os.close(descriptor)
        path = Path(text_path)
        path.write_text(text, encoding="utf-8")
        try:
            process = await asyncio.create_subprocess_exec(
                self.binary_path,
                "-m",
                self.model_path,
                "-f",
                str(path),
                "-c",
                str(self.max_context_length),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await process.communicate()
            output = stdout.decode("utf-8", errors="replace")
            if process.returncode != 0:
                raise RuntimeError(
                    f"llama-perplexity exited with code {process.returncode}: "
                    f"{output[-500:]}"
                )
            matches = self._PPL_PATTERN.findall(output)
            if not matches:
                raise RuntimeError("llama-perplexity output contains no final PPL")
            value = float(matches[-1])
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("llama-perplexity returned an invalid PPL")
            return value
        finally:
            path.unlink(missing_ok=True)

    async def close(self) -> None:
        return None


def _source_value(source: ModelSource | ArtifactReference) -> str:
    return source.repo_id or str(Path(source.local_path or "").expanduser())


async def build_evaluator(
    source: ModelSource | ArtifactReference,
    *,
    gguf_context_length: int = 2048,
) -> PerplexityEvaluator:
    value = _source_value(source)
    declared = getattr(source, "format", None)
    if value.lower().endswith(".gguf") or (
        isinstance(declared, str) and declared.lower() == "gguf"
    ):
        return LlamaPerplexityEvaluator(
            value,
            max_context_length=gguf_context_length,
        )
    return await asyncio.to_thread(
        TransformersEvaluator,
        value,
        source.revision,
    )


def _validation_texts(
    policy: ValidationPolicy,
    context_length: int,
) -> dict[str, list[str]]:
    path = Path(__file__).resolve().parents[1] / "config" / "validation_suite.json"
    suite = json.loads(path.read_text(encoding="utf-8"))
    texts: dict[str, list[str]] = {}
    for domain in policy.domains:
        if domain == "retrieval":
            texts[domain] = [
                generate_retrieval_prompt(context_length, context_length)
            ]
        else:
            texts[domain] = [
                f"{entry['prompt']}\nReference answer:\n{entry['expected_output']}"
                for entry in suite[domain]
            ]
    return texts


async def validate_reference_candidate(
    reference: PerplexityEvaluator,
    candidate: PerplexityEvaluator,
    *,
    policy: ValidationPolicy,
    golden_prompts: list[ValidationPrompt] | None = None,
) -> ValidationResult:
    """Compute strict three-domain or policy-selected PPL deltas."""

    context_length = min(
        reference.max_context_length,
        candidate.max_context_length,
        4096,
    )
    texts = _validation_texts(policy, context_length)
    suite_path = Path(__file__).resolve().parents[1] / "config" / "validation_suite.json"
    weights = json.loads(suite_path.read_text(encoding="utf-8"))["default_weights"]
    selected_weight = sum(float(weights[domain]) for domain in texts)
    per_domain: dict[str, DomainValidationResult] = {}
    for domain, domain_texts in texts.items():
        reference_values = [
            await reference.perplexity(text) for text in domain_texts
        ]
        candidate_values = [
            await candidate.perplexity(text) for text in domain_texts
        ]
        original = sum(reference_values) / len(reference_values)
        quantized = sum(candidate_values) / len(candidate_values)
        per_domain[domain] = DomainValidationResult(
            original_perplexity=original,
            quantized_perplexity=quantized,
            delta=quantized - original,
        )
    composite = sum(
        result.delta * (float(weights[domain]) / selected_weight)
        for domain, result in per_domain.items()
    )
    severity = get_severity_tier(composite)
    golden_results: list[GoldenValidationResult] = []
    for prompt in golden_prompts or []:
        text = (
            f"{prompt.prompt}\nReference answer:\n{prompt.expected_output}"
            if prompt.expected_output
            else prompt.prompt
        )
        delta = (
            await candidate.perplexity(text)
            - await reference.perplexity(text)
        )
        golden_results.append(
            GoldenValidationResult(
                prompt=prompt.prompt,
                expected_output=prompt.expected_output,
                actual_output=f"perplexity_delta={delta:.6f}",
                passed=delta <= 0.15,
            )
        )
    return ValidationResult(
        per_domain=per_domain,
        composite_delta=composite,
        severity_tier=severity["severity_tier"],
        requires_confirmation=severity["requires_confirmation"],
        quarantined=severity["quarantined"],
        golden_results=golden_results,
    )
