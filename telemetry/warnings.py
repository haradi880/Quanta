"""Fast, in-memory telemetry threshold evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TelemetryAlert:
    metric: str
    level: Literal["warning", "critical", "emergency"]
    value: float
    threshold: float
    system_action: str


_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "telemetry_thresholds.json"
)
with _CONFIG_PATH.open(encoding="utf-8") as _threshold_file:
    _THRESHOLDS = json.load(_threshold_file)["metrics"]

_LEVELS = ("emergency", "critical", "warning")


def evaluate_tick(metrics_dict: dict[str, Any]) -> list[TelemetryAlert]:
    """Return the highest active alert per metric without performing I/O."""

    alerts: list[TelemetryAlert] = []
    for metric, policy in _THRESHOLDS.items():
        if metric not in metrics_dict:
            continue
        try:
            value = float(metrics_dict[metric])
        except (TypeError, ValueError):
            continue
        comparison = policy["comparison"]
        for level in _LEVELS:
            rule = policy[level]
            threshold = rule.get("threshold")
            if rule.get("enabled", True) is False or threshold is None:
                continue
            duration_required = rule.get("duration_seconds")
            if duration_required is not None:
                duration = float(
                    metrics_dict.get(f"{metric}_duration_seconds", 0)
                )
                if duration < duration_required:
                    continue
            breached = (
                value > float(threshold)
                if comparison == "gt"
                else value < float(threshold)
            )
            if breached:
                alerts.append(
                    TelemetryAlert(
                        metric=metric,
                        level=level,
                        value=value,
                        threshold=float(threshold),
                        system_action=rule["system_action"],
                    )
                )
                break
    return alerts
