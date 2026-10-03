"""Detection metrics. F1 at IoU 0.5 with class-aware greedy matching, as in the paper."""

from collections.abc import Sequence
from dataclasses import dataclass

from discern.eval.types import GroundTruthBox
from discern.models.roles import Detection
from discern.vision.boxes import iou


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_image(
    predictions: Sequence[Detection],
    ground_truth: Sequence[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> Counts:
    """Greedy matching in descending score order. A prediction matches the unmatched
    ground-truth box of the same label with the highest IoU, if that IoU reaches the threshold."""
    matched: set[int] = set()
    tp = 0
    for pred in sorted(predictions, key=lambda d: d.score, reverse=True):
        best_idx, best_iou = -1, iou_threshold
        for i, gt in enumerate(ground_truth):
            if i in matched or gt.label != pred.label:
                continue
            overlap = iou(pred.box, gt.box)
            if overlap >= best_iou:
                best_idx, best_iou = i, overlap
        if best_idx >= 0:
            matched.add(best_idx)
            tp += 1
    return Counts(tp=tp, fp=len(predictions) - tp, fn=len(ground_truth) - tp)
