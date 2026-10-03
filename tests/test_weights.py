import hashlib
from pathlib import Path

import pytest

from discern.models.weights import ChecksumError, WeightSpec, ensure_weights, sha256_of

PAYLOAD = b"tiny fake weights"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _source(tmp_path: Path) -> Path:
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    return src


def test_sha256_of(tmp_path: Path) -> None:
    assert sha256_of(_source(tmp_path)) == DIGEST


def test_downloads_missing_file_and_verifies(tmp_path: Path) -> None:
    spec = WeightSpec("w.bin", _source(tmp_path).as_uri(), DIGEST)
    path = ensure_weights(spec, tmp_path / "weights")
    assert path.read_bytes() == PAYLOAD
    assert not list((tmp_path / "weights").glob("*.part"))


def test_bad_download_is_rejected_and_not_kept(tmp_path: Path) -> None:
    spec = WeightSpec("w.bin", _source(tmp_path).as_uri(), "0" * 64)
    with pytest.raises(ChecksumError):
        ensure_weights(spec, tmp_path / "weights")
    assert not list((tmp_path / "weights").iterdir())


def test_existing_good_file_is_not_redownloaded(tmp_path: Path) -> None:
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "w.bin").write_bytes(PAYLOAD)
    spec = WeightSpec("w.bin", "file:///does/not/exist", DIGEST)
    assert ensure_weights(spec, directory) == directory / "w.bin"


def test_existing_corrupt_file_raises(tmp_path: Path) -> None:
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "w.bin").write_bytes(b"corrupt")
    with pytest.raises(ChecksumError):
        ensure_weights(WeightSpec("w.bin", "file:///x", DIGEST), directory)
