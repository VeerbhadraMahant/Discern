"""Evaluation gate (system-design 10.3): compare measured metrics with the configured thresholds."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from discern.config.settings import EvalGateThresholds


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]  # one line per failed check; empty when the gate passes


def evaluate_gate(metrics: Mapping[str, float], gate: EvalGateThresholds) -> GateResult:
    """A metric with a minimum must be at least `minimum * (1 - tolerance)`; one with a maximum
    must be at most `maximum * (1 + tolerance)`. A gated metric that is missing or NaN fails."""
    reasons: list[str] = []
    checks = [(n, v, True) for n, v in gate.minimums.items()]
    checks += [(n, v, False) for n, v in gate.maximums.items()]
    for name, bound, is_minimum in checks:
        value = metrics.get(name)
        if value is None or math.isnan(value):
            reasons.append(f"{name}: no value measured")
        elif is_minimum:
            floor = bound * (1.0 - gate.tolerance)
            if value < floor:
                reasons.append(f"{name}: {value:.4g} is below the minimum {floor:.4g}")
        else:
            ceiling = bound * (1.0 + gate.tolerance)
            if value > ceiling:
                reasons.append(f"{name}: {value:.4g} is above the maximum {ceiling:.4g}")
    return GateResult(passed=not reasons, reasons=tuple(reasons))
