import torch

from discern.models.adapters._torch_io import to_image, to_tensor
from discern.models.adapters._vendor.mprnet.mprnet import MPRNet
from discern.models.manager import RegistryEntry
from discern.models.roles import Image
from discern.models.tiling import pad_to_multiple, run_tiled
from discern.models.weights import WeightSpec, ensure_weights

WEIGHTS = WeightSpec(
    filename="mprnet_derain.pth",
    url=(
        "https://drive.usercontent.google.com/download"
        "?id=1O3WEJbcat7eTY6doXWeorAbQ1l_WmMnM&export=download&confirm=t"
    ),
    sha256="d73d057a097e1cc97fbbea8481e98873f61ba4ee416834d9ef8201908e46dee5",
)
TILE = 512
TILE_PAD = 32
MULTIPLE = 8


class MprnetAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        net = MPRNet()
        state = torch.load(ensure_weights(WEIGHTS), map_location="cpu", weights_only=True)
        net.load_state_dict(state["state_dict"], strict=True)
        self._net = net.eval().to(device)

    @torch.inference_mode()
    def _derain_tile(self, tile: Image) -> Image:
        padded, (h, w) = pad_to_multiple(tile, MULTIPLE)
        out = self._net(to_tensor(padded, self._device))[0]  # first output is the final stage
        return to_image(out[:, :, :h, :w])

    def restore(self, image: Image) -> Image:
        return run_tiled(image, self._derain_tile, 1, TILE, TILE_PAD)
