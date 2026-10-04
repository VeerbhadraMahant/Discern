"""Pure helpers of the DetAS versus DetAS-X ablation (scripts/run_ablation_m9.py)."""

from collections import Counter
from collections.abc import Mapping, Sequence

from discern.agent.nodes.restorer_select import NONE
from discern.agent.schemas import ShotPlan
from discern.models.roles import Detection
from discern.vision.boxes import clip, scale

ARMS = ("detas", "detas_x", "detas_xp")  # no experience, experience text, experience policy
ARM_LABELS = {"detas": "DetAS", "detas_x": "DetAS-X", "detas_xp": "DetAS-XP"}
BASELINE = ARMS[0]
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


def plan_cache_name(
    dataset: str, vlm_tag: str, memory_version: str, arm: str = "detas_x"
) -> str:
    """Plan cache of an experience arm (the no-experience arm reuses the Milestone 2 cache). The
    experience-text arm keeps its original file name so earlier caches stay valid."""
    prefix = "plans" if arm == "detas_x" else f"plans-{arm}"
    return f"{prefix}-{dataset}-{vlm_tag}-{memory_version}.json"


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
    f1: Mapping[tuple[str, str, str], float],
    datasets: Sequence[str],
    metrics: Sequence[str],
    arms: Sequence[str] = ARMS,
) -> str:
    """Markdown table of F1 per dataset and metric: the baseline arm (first of `arms`), then each
    other arm with its delta against the baseline, and a mean row per metric. `f1` is keyed by
    (dataset, arm, metric); missing cells print as n/a. In the mean row an arm's delta is taken
    over the datasets where both it and the baseline have a value."""
    base, others = arms[0], arms[1:]
    header = [ARM_LABELS[base]] + [c for a in others for c in (ARM_LABELS[a], "delta")]
    lines = [
        "| dataset | metric | " + " | ".join(header) + " |",
        "|" + "---|" * (2 + len(header)),
    ]
    for metric in metrics:
        paired: dict[str, list[tuple[float, float]]] = {a: [] for a in others}
        bases: list[float] = []
        for ds in datasets:
            a = f1.get((ds, base, metric))
            cells = [_cell(a)]
            bases += [] if a is None else [a]
            for arm in others:
                b = f1.get((ds, arm, metric))
                if a is None or b is None:
                    cells += [_cell(b), "n/a"]
                    continue
                paired[arm].append((a, b))
                cells += [f"{b:.3f}", f"{b - a:+.3f}"]
            lines.append(f"| {ds} | {metric} | " + " | ".join(cells) + " |")
        if any(paired.values()):
            cells = [f"{sum(bases) / len(bases):.3f}"]  # bases is non-empty when any pair exists
            for arm in others:
                pairs = paired[arm]
                if not pairs:
                    cells += ["n/a", "n/a"]
                    continue
                ma, mb = (sum(p[i] for p in pairs) / len(pairs) for i in (0, 1))
                cells += [f"{mb:.3f}", f"{mb - ma:+.3f}"]
            lines.append(f"| mean | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
