import torch

from discern.models.adapters._torch_io import to_image, to_tensor
from discern.models.adapters._vendor.real_esrgan.rrdbnet import RRDBNet
from discern.models.manager import RegistryEntry
from discern.models.roles import Image
from discern.models.tiling import run_tiled, upscale_from_x4
from discern.models.weights import WeightSpec, ensure_weights

WEIGHTS = WeightSpec(
    filename="RealESRGAN_x4plus.pth",
    url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
)
TILE = 256
TILE_PAD = 16


class RealEsrganAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        self._half = device.startswith("cuda")
        net = RRDBNet()
        state = torch.load(ensure_weights(WEIGHTS), map_location="cpu", weights_only=True)
        net.load_state_dict(state["params_ema"], strict=True)
        net = net.eval().to(device)
        self._net = net.half() if self._half else net

    @torch.inference_mode()
    def _x4_tile(self, tile: Image) -> Image:
        x = to_tensor(tile, self._device)
        out = self._net(x.half() if self._half else x)
        return to_image(out.float())

    def upscale(self, image: Image, factor: int) -> Image:
        return upscale_from_x4(
            image, factor, lambda im: run_tiled(im, self._x4_tile, 4, TILE, TILE_PAD)
        )
