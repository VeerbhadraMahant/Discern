import re
from collections.abc import Sequence

import torch
from PIL import Image as PILImage
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from discern.models.manager import RegistryEntry
from discern.models.roles import SCORE_FLOOR, Detection, Image
from discern.vision.boxes import Box, clip


def _match_target(phrase: str, targets: Sequence[str]) -> str | None:
    """Grounding DINO returns text spans; map one back to a requested target."""
    phrase = phrase.strip().lower()
    for t in targets:
        if phrase == t.lower():
            return t
    for t in targets:
        if re.search(rf"\b{re.escape(t.lower())}\b", phrase):
            return t
    return None


class GroundingDinoAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        self._processor = AutoProcessor.from_pretrained(entry.model_id, revision=entry.revision)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                entry.model_id, revision=entry.revision
            )
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        h, w = image.shape[:2]
        lowered = [t.lower() for t in targets]
        text = " ".join(f"{t}." for t in lowered)
        inputs = self._processor(
            images=PILImage.fromarray(image), text=text, return_tensors="pt"
        ).to(self._device)
        outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=SCORE_FLOOR,
            text_threshold=SCORE_FLOOR,
            target_sizes=[(h, w)],
        )[0]
        detections = []
        for box, score, phrase in zip(
            result["boxes"], result["scores"], result["text_labels"], strict=True
        ):
            label = _match_target(phrase, targets)
            if label is not None:
                detections.append(
                    Detection(
                        box=clip(Box(*box.tolist()), w, h),
                        label=label,
                        score=float(score),
                        detector=self.name,
                    )
                )
        return detections
