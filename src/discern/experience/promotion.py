"""Promotion of a candidate memory version (system-design 5.7 online growth).

The candidate is promoted only when it passes the evaluation gate and does not regress against the
currently pinned baseline on any gated metric (beyond the gate tolerance). Promotion writes the
pinned memory version id into a small JSON pointer file that the app reads.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from discern.config.settings import EvalGateThresholds
from discern.eval.gate import evaluate_gate


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]  # one line per failed check; empty when promoted


def promote_if_passes(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    gate_thresholds: EvalGateThresholds,
) -> PromotionDecision:
    """Gate the candidate, then require it not to fall behind the baseline by more than the gate
    tolerance (relative) on every gated metric the baseline has a value for."""
    reasons = list(evaluate_gate(candidate_metrics, gate_thresholds).reasons)
    tolerance = gate_thresholds.tolerance
    gated = [(n, True) for n in gate_thresholds.minimums] + [
        (n, False) for n in gate_thresholds.maximums
    ]
    for name, higher_is_better in gated:
        cand, base = candidate_metrics.get(name), baseline_metrics.get(name)
        if cand is None or base is None or math.isnan(cand) or math.isnan(base):
            continue  # a missing candidate value is already reported by the gate
        if higher_is_better and cand < base * (1.0 - tolerance):
            reasons.append(f"{name}: {cand:.4g} regresses from the baseline {base:.4g}")
        elif not higher_is_better and cand > base * (1.0 + tolerance):
            reasons.append(f"{name}: {cand:.4g} regresses from the baseline {base:.4g}")
    return PromotionDecision(promote=not reasons, reasons=tuple(reasons))


def write_pointer(path: Path, version_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"memory_version": version_id}) + "\n", encoding="utf-8")


def read_pointer(path: Path) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("memory_version") if isinstance(data, dict) else None
    return version if isinstance(version, str) else None


def promote_candidate(
    version_id: str,
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    gate_thresholds: EvalGateThresholds,
    pointer_path: Path,
) -> PromotionDecision:
    """Decide, and on a pass pin `version_id` in the pointer file. A failure leaves it untouched."""
    decision = promote_if_passes(candidate_metrics, baseline_metrics, gate_thresholds)
    if decision.promote:
        write_pointer(pointer_path, version_id)
    return decision
