from collections.abc import Sequence

import torch
from PIL import Image as PILImage
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from discern.models.manager import RegistryEntry
from discern.models.roles import SCORE_FLOOR, Detection, Image
from discern.vision.boxes import Box, clip


class Owlv2Adapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        self._processor = Owlv2Processor.from_pretrained(entry.model_id, revision=entry.revision)
        self._model = (
            Owlv2ForObjectDetection.from_pretrained(entry.model_id, revision=entry.revision)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        h, w = image.shape[:2]
        queries = [f"a photo of a {t}" for t in targets]
        inputs = self._processor(
            text=[queries], images=PILImage.fromarray(image), return_tensors="pt"
        ).to(self._device)
        outputs = self._model(**inputs)
        # OWLv2 pads to a square, so boxes are relative to the padded side.
        side = max(h, w)
        result = self._processor.post_process_object_detection(
            outputs, threshold=SCORE_FLOOR, target_sizes=torch.tensor([[side, side]])
        )[0]
        return [
            Detection(
                box=clip(Box(*box.tolist()), w, h),
                label=targets[int(label)],
                score=float(score),
                detector=self.name,
            )
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"], strict=True
            )
        ]
