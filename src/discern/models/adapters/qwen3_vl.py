from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from discern.models.grounding_parse import parse_grounding
from discern.models.manager import RegistryEntry
from discern.models.roles import Detection, Image

MAX_NEW_TOKENS = 768
# Cap visual tokens (28x28 px patches after merging) so 8GB VRAM is enough for large frames.
MAX_PIXELS = 1280 * 28 * 28

_DETECT_PROMPT = (
    "Locate every instance of the following categories in the image: {targets}. "
    "Reply with only a JSON list, one object per instance, in the form "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "<category>"}}. '
    "Use one of the listed category names as the label. If none are present, reply with []."
)


class Qwen3VLAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            entry.model_id, revision=entry.revision, max_pixels=MAX_PIXELS
        )
        precision: dict[str, Any] = (
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
            if entry.quantize_4bit
            else {"dtype": torch.bfloat16}
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            entry.model_id,
            revision=entry.revision,
            device_map={"": device},
            **precision,
        ).eval()

    @torch.inference_mode()
    def generate(self, prompt: str, images: Sequence[Image] = ()) -> str:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[PILImage.fromarray(i) for i in images] or None,
            return_tensors="pt",
        ).to(self._device)
        out = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return str(
            self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
        )

    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        h, w = image.shape[:2]
        reply = self.generate(_DETECT_PROMPT.format(targets=", ".join(targets)), [image])
        return parse_grounding(reply, w, h, targets, self.name)
