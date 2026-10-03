import cv2
import numpy as np


def low_light_enhancement(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gamma = 1.6
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(out, table)


def dehaze(img: np.ndarray) -> np.ndarray:
    # Simplified dark-channel-prior dehazing.
    dark = _dark_channel(img)
    atmosphere = _atmospheric_light(img, dark)
    transmission = _transmission(img, atmosphere)
    return _recover(img, atmosphere, transmission)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)


def _dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_channel, kernel)


def _atmospheric_light(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    flat_dark = dark.flatten()
    flat_img = img.reshape(-1, 3)
    n_pixels = max(int(flat_dark.size * 0.001), 1)
    idx = np.argsort(flat_dark)[-n_pixels:]
    return np.max(flat_img[idx], axis=0).astype(np.float64)


def _transmission(img: np.ndarray, atmosphere: np.ndarray, omega: float = 0.85) -> np.ndarray:
    normalized = img.astype(np.float64) / np.clip(atmosphere, 1, 255)
    return 1 - omega * _dark_channel((normalized * 255).astype(np.uint8)) / 255.0


def _recover(img: np.ndarray, atmosphere: np.ndarray, transmission: np.ndarray, t_min: float = 0.1) -> np.ndarray:
    t = np.clip(transmission, t_min, 1.0)[:, :, np.newaxis]
    result = (img.astype(np.float64) - atmosphere) / t + atmosphere
    return np.clip(result, 0, 255).astype(np.uint8)


RESTORATION_FUNCS = {
    "low_light_enhancement": low_light_enhancement,
    "dehaze": dehaze,
    "denoise": denoise,
}
