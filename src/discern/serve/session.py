"""Per-session directories. Uploaded media lives only here and is deleted on TTL expiry.

Session ids come from `secrets.token_urlsafe` and are validated on every use. Uploads are stored
under generated names; the uploader's file name is used only to read the extension.
"""

import re
import secrets
import shutil
import time
from collections.abc import Callable
from pathlib import Path

ALLOWED_EXTENSIONS = frozenset({"mp4", "mov", "webm", "jpg", "jpeg", "png", "webp"})
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,64}$")  # token_urlsafe(32) is 43 characters
_ID_BYTES = 32
_MARKER = ".touched"
_BYTES_PER_MB = 1024 * 1024


class UploadRejected(ValueError):
    """The upload failed validation (extension or size)."""


def validate_upload(filename: str, size_bytes: int, max_upload_mb: float) -> str:
    """Return the normalised extension, or raise `UploadRejected`. Only the final suffix counts,
    so a name such as `a.png/../b.exe` is judged by `exe` and rejected."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadRejected(f"file type {extension!r} is not allowed")
    if size_bytes <= 0:
        raise UploadRejected("the upload is empty")
    if size_bytes > max_upload_mb * _BYTES_PER_MB:
        raise UploadRejected(f"the upload exceeds {max_upload_mb:g} MB")
    return extension


class SessionStore:
    def __init__(
        self, root: Path, ttl_seconds: float, clock: Callable[[], float] = time.time
    ) -> None:
        self.root = root
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> str:
        session_id = secrets.token_urlsafe(_ID_BYTES)
        # Build under a name the sweeper ignores, then rename: a sweep never sees a marker-less
        # (hence "expired") new session.
        staging = self.root / f".new-{session_id}"
        staging.mkdir()
        (staging / _MARKER).write_text(repr(self._clock()), encoding="utf-8")
        staging.rename(self.root / session_id)
        return session_id

    def path(self, session_id: str) -> Path:
        """Directory of an existing session. Raises `KeyError` for a malformed or unknown id."""
        if not _ID_PATTERN.fullmatch(session_id):
            raise KeyError("unknown session")
        directory = self.root / session_id
        if directory.is_symlink() or not directory.is_dir():
            raise KeyError("unknown session")
        return directory

    def touch(self, session_id: str) -> None:
        marker = self.path(session_id) / _MARKER
        marker.write_text(repr(self._clock()), encoding="utf-8")

    def save_upload(
        self, session_id: str, filename: str, data: bytes, max_upload_mb: float
    ) -> Path:
        """Validate and store an upload under a generated name; returns the stored path."""
        extension = validate_upload(filename, len(data), max_upload_mb)
        directory = self.path(session_id)
        stored = directory / f"upload-{secrets.token_hex(8)}.{extension}"
        stored.write_bytes(data)
        self.touch(session_id)
        return stored

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self.path(session_id))

    def cleanup_expired(self, now: float | None = None) -> list[str]:
        """Delete sessions untouched for more than `ttl_seconds`; returns the deleted ids.

        Only real directories directly under the root with a well-formed id are considered, so
        nothing outside the root (or behind a symlink) is ever removed. A directory with a missing
        or unreadable marker counts as expired.
        """
        current = self._clock() if now is None else now
        deleted: list[str] = []
        for child in sorted(self.root.iterdir()):
            if not _ID_PATTERN.fullmatch(child.name) or child.is_symlink() or not child.is_dir():
                continue
            if current - self._last_touched(child) > self.ttl_seconds:
                try:
                    shutil.rmtree(child)
                except OSError:  # for example a file still open on Windows: retry next sweep
                    continue
                deleted.append(child.name)
        return deleted

    @staticmethod
    def _last_touched(directory: Path) -> float:
        try:
            return float((directory / _MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return float("-inf")
