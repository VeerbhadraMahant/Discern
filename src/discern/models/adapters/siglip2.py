from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoModel, AutoProcessor

from discern.models.manager import RegistryEntry
from discern.models.roles import Embeddings, Image

TEXT_MAX_LENGTH = 64  # SigLIP text towers are trained with fixed-length padding


class Siglip2Adapter:
    """Embedder role: L2-normalised SigLIP 2 image and text embeddings."""

    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        self._processor = AutoProcessor.from_pretrained(entry.model_id, revision=entry.revision)
        self._model = (
            AutoModel.from_pretrained(entry.model_id, revision=entry.revision).to(device).eval()
        )

    @staticmethod
    def _normalise(output: object) -> Embeddings:
        # transformers 5 returns a model-output object whose pooler_output is the embedding.
        features: torch.Tensor = getattr(output, "pooler_output", output)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image]) -> Embeddings:
        inputs = self._processor(
            images=[PILImage.fromarray(i) for i in images], return_tensors="pt"
        ).to(self._device)
        return self._normalise(self._model.get_image_features(**inputs))

    @torch.inference_mode()
    def embed_text(self, texts: Sequence[str]) -> Embeddings:
        inputs = self._processor(
            text=[t.lower() for t in texts],
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        return self._normalise(self._model.get_text_features(**inputs))
