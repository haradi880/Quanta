import math

import pytest

from core.accelerator import (
    calc_perplexity,
    get_severity_tier,
    run_validation_suite,
)


class UniformModel:
    max_context_length = 256

    def tokenize(self, text):
        return list(range(max(2, len(text.split()))))

    def log_probabilities(self, tokens):
        return [-math.log(10.0)] * (len(tokens) - 1)


def test_perplexity_uses_causal_prediction_count():
    class TotalLikelihoodModel:
        def log_likelihood(self, tokens):
            return -(len(tokens) - 1) * math.log(10.0)

    assert calc_perplexity(TotalLikelihoodModel(), [1, 2, 3, 4]) == pytest.approx(10)


def test_identical_models_have_zero_validation_delta():
    result = run_validation_suite(
        UniformModel(),
        UniformModel(),
        [{"prompt": "What is two plus two?", "expected_output": "4"}],
    )

    assert result.composite_delta == pytest.approx(0)
    assert result.severity_tier == "excellent"
    assert result.golden_results[0].passed is True


def test_poor_severity_requires_confirmation():
    result = get_severity_tier(0.50)

    assert result["severity_tier"] == "poor"
    assert result["requires_confirmation"] is True
    assert result["quarantined"] is False


def test_critical_severity_is_quarantined():
    result = get_severity_tier(0.61)

    assert result["severity_tier"] == "critical"
    assert result["quarantined"] is True
