import torch

from discern.models.adapters._torch_io import to_image, to_tensor
from discern.models.adapters._vendor.zero_dce.model import enhance_net_nopool
from discern.models.manager import RegistryEntry
from discern.models.roles import Image
from discern.models.tiling import pad_to_multiple
from discern.models.weights import WeightSpec, ensure_weights

WEIGHTS = WeightSpec(
    filename="Epoch99.pth",
    url=(
        "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/"
        "09f202b690f82da939b8e6ec8535960ae97ad8bd/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
    ),
    sha256="ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84",
)
SCALE_FACTOR = 12  # upstream default; the curve map is estimated at 1/12 resolution


class ZeroDcePpAdapter:
    def __init__(self, entry: RegistryEntry, device: str = "cuda") -> None:
        self.name = entry.name
        self._device = device
        net = enhance_net_nopool(SCALE_FACTOR)
        net.load_state_dict(
            torch.load(ensure_weights(WEIGHTS), map_location="cpu", weights_only=True),
            strict=True,
        )
        self._net = net.eval().to(device)

    @torch.inference_mode()
    def restore(self, image: Image) -> Image:
        padded, (h, w) = pad_to_multiple(image, SCALE_FACTOR)
        enhanced, _ = self._net(to_tensor(padded, self._device))
        return to_image(enhanced[:, :, :h, :w])
