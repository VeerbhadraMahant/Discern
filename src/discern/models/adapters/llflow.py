import cv2
import numpy as np
import torch

from discern.models.adapters._torch_io import to_image, to_tensor
from discern.models.adapters._vendor.llflow._util import to_none_dict
from discern.models.adapters._vendor.llflow.LLFlow_arch import LLFlow
from discern.models.manager import RegistryEntry
from discern.models.roles import Image
from discern.models.tiling import pad_to_multiple
from discern.models.weights import WeightSpec, ensure_weights

# Small LOL model; weights hosted on upstream's public Google Drive link (no login needed).
WEIGHTS = WeightSpec(
    filename="llflow_lol_small.pth",
    url=(
        "https://drive.usercontent.google.com/download"
        "?id=1tukKu2KBZ_ohlQiLG4EKnrn1CAt_2F6G&export=download&confirm=t"
    ),
    sha256="2bf4c9192b401bf7155b2aa0781d9d8eed2e0bcc148286a9e2b224e12777bb38",
)
MULTIPLE = 16

# Upstream code/confs/LOL_smallNet.yml, reduced to the keys the network reads.
_OPT = {
    "scale": 1,
    "cond_encoder": "ConEncoder1",
    "concat_histeq": True,
    "concat_color_map": False,
    "gray_map": False,
    "encode_color_map": False,
    "to_yuv": False,
    "align_maxpool": True,
    "le_curve": False,
    "datasets": {"train": {"GT_size": 160, "quant": 32, "log_low": True}},
    "network_G": {
        "in_nc": 3,
        "out_nc": 3,
        "nf": 32,
        "nb": 4,
        "flow": {
            "K": 4,
            "L": 3,
            "noInitialInj": True,
            "coupling": "CondAffineSeparatedAndCond",
            "additionalFlowNoAffine": 2,
            "conditionInFeaDim": 64,
            "split": {"enable": False},
            "fea_up0": True,
            "stackRRDB": {"blocks": [1], "concat": True},
        },
    },
}


class LlflowAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        opt = to_none_dict(_OPT)
        net = LLFlow(3, 3, nf=32, nb=4, gc=32, scale=1, K=4, opt=opt)
        state = torch.load(ensure_weights(WEIGHTS), map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        self._net = net.eval().to(device)

    @torch.inference_mode()
    def restore(self, image: Image) -> Image:
        padded, (h, w) = pad_to_multiple(image, MULTIPLE)
        # Conditioning input as in upstream test_unpaired.py: log of the low-light image
        # concatenated with its per-channel histogram equalisation.
        his = np.stack(
            [cv2.equalizeHist(np.ascontiguousarray(padded[:, :, c])) for c in range(3)], 2
        )
        low = torch.log(torch.clamp(to_tensor(padded, self._device) + 1e-3, min=1e-3))
        lq = torch.cat([low, to_tensor(his, self._device)], dim=1)
        with torch.autocast(self._device.split(":")[0], enabled=self._device.startswith("cuda")):
            out, _ = self._net(lr=lq, z=None, eps_std=0, reverse=True)
        return to_image(out.float()[:, :, :h, :w])
