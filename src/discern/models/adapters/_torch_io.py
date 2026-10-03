"""numpy <-> torch conversion shared by the restoration adapters."""

import numpy as np
import torch

from discern.models.roles import Image


def to_tensor(image: Image, device: str) -> torch.Tensor:
    return torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def to_image(tensor: torch.Tensor) -> Image:
    arr = tensor.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(arr)
