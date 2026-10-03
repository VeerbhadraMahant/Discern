"""Weight files for adapters: pinned URL and SHA256, downloaded on first use to data/_weights/."""

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from discern.config.settings import CONFIG_DIR

WEIGHTS_DIR = CONFIG_DIR.parent / "data" / "_weights"
_CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightSpec:
    filename: str
    url: str
    sha256: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(spec: WeightSpec, directory: Path = WEIGHTS_DIR) -> Path:
    """Return the verified local path of `spec`, downloading it first if it is missing.

    A file that exists but fails verification is never used and never silently replaced.
    """
    path = directory / spec.filename
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(spec.url) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
        if sha256_of(partial) != spec.sha256:
            partial.unlink()
            raise ChecksumError(f"{spec.filename}: downloaded file does not match pinned SHA256")
        partial.replace(path)
        return path
    if sha256_of(path) != spec.sha256:
        raise ChecksumError(f"{path} does not match pinned SHA256; delete it to re-download")
    return path
