import torch

from discern.models.adapters._torch_io import to_image, to_tensor
from discern.models.adapters._vendor.swinir.network_swinir import SwinIR
from discern.models.manager import RegistryEntry
from discern.models.roles import Image
from discern.models.tiling import run_tiled
from discern.models.weights import WeightSpec, ensure_weights

WEIGHTS = WeightSpec(
    filename="005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    url=(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"
    ),
    sha256="917cd972f7ba80786871add249ad43e4477ce2db59b4ad63e2fa446f7221d013",
)
TILE = 384
TILE_PAD = 16


class SwinirAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        net = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6] * 6,
            embed_dim=180,
            num_heads=[6] * 6,
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        state = torch.load(ensure_weights(WEIGHTS), map_location="cpu", weights_only=True)
        net.load_state_dict(state["params"], strict=True)
        self._net = net.eval().to(device)

    @torch.inference_mode()
    def _denoise_tile(self, tile: Image) -> Image:
        # SwinIR.forward pads to a multiple of the window size and crops back itself.
        return to_image(self._net(to_tensor(tile, self._device)))

    def restore(self, image: Image) -> Image:
        return run_tiled(image, self._denoise_tile, 1, TILE, TILE_PAD)
