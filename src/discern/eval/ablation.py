"""Pure helpers of the DetAS versus DetAS-X ablation (scripts/run_ablation_m9.py)."""

from collections import Counter
from collections.abc import Mapping, Sequence

from discern.agent.nodes.restorer_select import NONE
from discern.agent.schemas import ShotPlan
from discern.models.roles import Detection
from discern.vision.boxes import clip, scale

ARMS = ("detas", "detas_x")
NO_MEMORY = "none"


def rescale_detections(
    detections: Sequence[Detection], factor: int, width: int, height: int
) -> list[Detection]:
    """Map boxes found on a `factor`-times super-resolved image back to the original
    `width` x `height` frame and clip them to it."""
    return [
        d.model_copy(update={"box": clip(scale(d.box, 1.0 / factor), width, height)})
        for d in detections
    ]


def detection_cache_name(
    arm: str, detector: str, dataset: str, revision: str, vlm_tag: str, memory_version: str
) -> str:
    """Cache file of one arm's raw detections. The detections depend on the SAIR plans (VLM tag and,
    for the experience arm, the memory version) and on the detector revision; the `-sr` suffix
    marks detections made on super-resolved inputs, so they never mix with older caches."""
    return f"{arm}-{detector}-{dataset}-{revision[:10]}-{vlm_tag}-{memory_version}-sr.json"


def plan_cache_name(dataset: str, vlm_tag: str, memory_version: str) -> str:
    """Plan cache of the experience arm (the no-experience arm reuses the Milestone 2 cache)."""
    return f"plans-{dataset}-{vlm_tag}-{memory_version}.json"


def decision_stats(
    plans: Sequence[ShotPlan], predicted: Sequence[str], truth: Sequence[str | None]
) -> dict[str, float]:
    """Rates of restoring, accepting the restored image, and applying SR, plus perception accuracy
    against the dataset-implied labels."""
    n = len(plans)
    if n == 0:
        return {"restore_rate": 0.0, "accept_rate": 0.0, "sr_rate": 0.0, "perception_acc": 0.0}
    hits = sum(p == t for p, t in zip(predicted, truth, strict=True))
    return {
        "restore_rate": sum(p.restorer != NONE for p in plans) / n,
        "accept_rate": sum(p.use_restored for p in plans) / n,
        "sr_rate": sum(p.sr_factor is not None for p in plans) / n,
        "perception_acc": hits / n,
    }


def pair_frequency(pairs: Sequence[Sequence[str]]) -> dict[str, int]:
    """How often each detector set was chosen, e.g. {'owlv2-base+yolo-world-v2': 7}."""
    return dict(Counter("+".join(sorted(p)) for p in pairs).most_common())


def delta_table(
    f1: Mapping[tuple[str, str, str], float], datasets: Sequence[str], metrics: Sequence[str]
) -> str:
    """Markdown table of DetAS versus DetAS-X F1 per dataset and metric, with the delta and a mean
    row per metric. `f1` is keyed by (dataset, arm, metric); missing cells print as n/a."""
    lines = ["| dataset | metric | DetAS | DetAS-X | delta |", "|---|---|---|---|---|"]
    for metric in metrics:
        pairs: list[tuple[float, float]] = []
        for ds in datasets:
            a, b = f1.get((ds, "detas", metric)), f1.get((ds, "detas_x", metric))
            if a is None or b is None:
                lines.append(f"| {ds} | {metric} | {_cell(a)} | {_cell(b)} | n/a |")
                continue
            pairs.append((a, b))
            lines.append(f"| {ds} | {metric} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        if pairs:
            ma, mb = (sum(p[i] for p in pairs) / len(pairs) for i in (0, 1))
            lines.append(f"| mean | {metric} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} |")
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
