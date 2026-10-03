from collections.abc import Sequence

from ultralytics import YOLOWorld

from discern.models.manager import RegistryEntry
from discern.models.roles import SCORE_FLOOR, Detection, Image
from discern.vision.boxes import Box


class YoloWorldAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        self._model = YOLOWorld(entry.model_id)
        self._targets: tuple[str, ...] = ()

    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        if tuple(targets) != self._targets:  # set_classes recomputes text embeddings
            self._model.set_classes(list(targets))
            self._targets = tuple(targets)
        # Ultralytics treats numpy input as BGR; our images are RGB.
        result = self._model.predict(
            image[:, :, ::-1], conf=SCORE_FLOOR, device=self._device, verbose=False
        )[0]
        return [
            Detection(
                box=Box(*b.xyxy[0].tolist()),
                label=self._targets[int(b.cls[0])],
                score=float(b.conf[0]),
                detector=self.name,
            )
            for b in result.boxes
        ]
