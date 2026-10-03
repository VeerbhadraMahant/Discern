from pathlib import Path

import pytest

from discern.serve.session import SessionStore, UploadRejected, validate_upload


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(tmp_path: Path, clock: Clock, ttl: float = 100.0) -> SessionStore:
    return SessionStore(tmp_path / "sessions", ttl, clock)


def test_ids_are_unique_and_unguessable(tmp_path: Path) -> None:
    store = make(tmp_path, Clock())
    ids = {store.create() for _ in range(20)}
    assert len(ids) == 20
    assert all(len(i) >= 43 for i in ids)
    assert all(store.path(i).parent == store.root for i in ids)


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "", ".", "x" * 43 + "/..", "short"])
def test_malformed_ids_are_unknown(tmp_path: Path, bad: str) -> None:
    store = make(tmp_path, Clock())
    with pytest.raises(KeyError):
        store.path(bad)


def test_upload_uses_generated_name(tmp_path: Path) -> None:
    store = make(tmp_path, Clock())
    sid = store.create()
    stored = store.save_upload(sid, "../../evil/My Photo.PNG", b"abc", 1)
    assert stored.parent == store.path(sid)
    assert stored.suffix == ".png" and "evil" not in stored.name and "Photo" not in stored.name
    assert stored.read_bytes() == b"abc"
    assert not (tmp_path / "evil").exists()


@pytest.mark.parametrize("name", ["a.exe", "a", "a.png.exe", "a.png/../b.sh", "..\\x.bat"])
def test_bad_extensions_rejected(name: str) -> None:
    with pytest.raises(UploadRejected):
        validate_upload(name, 10, 1)


@pytest.mark.parametrize("ext", ["mp4", "mov", "webm", "jpg", "jpeg", "png", "webp"])
def test_allowed_extensions(ext: str) -> None:
    assert validate_upload(f"clip.{ext.upper()}", 10, 1) == ext


def test_size_limits() -> None:
    assert validate_upload("a.png", 1024 * 1024, 1) == "png"
    with pytest.raises(UploadRejected):
        validate_upload("a.png", 1024 * 1024 + 1, 1)
    with pytest.raises(UploadRejected):
        validate_upload("a.png", 0, 1)


def test_expiry_and_touch(tmp_path: Path) -> None:
    clock = Clock()
    store = make(tmp_path, clock, ttl=100)
    old, fresh = store.create(), store.create()
    store.save_upload(old, "a.png", b"x", 1)
    clock.now += 60
    store.touch(fresh)
    clock.now += 60  # old is 120 s idle, fresh 60 s
    assert store.cleanup_expired() == [old]
    assert not (store.root / old).exists()
    assert store.path(fresh).is_dir()
    assert store.cleanup_expired(now=clock.now + 41) == [fresh]
    with pytest.raises(KeyError):
        store.path(fresh)


def test_exactly_at_ttl_is_kept(tmp_path: Path) -> None:
    clock = Clock()
    store = make(tmp_path, clock, ttl=100)
    sid = store.create()
    assert store.cleanup_expired(now=clock.now + 100) == []
    assert store.path(sid).is_dir()


def test_cleanup_never_touches_outside_root(tmp_path: Path) -> None:
    clock = Clock()
    store = make(tmp_path, clock, ttl=1)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    sibling = tmp_path / ("s" * 43)  # id-shaped directory beside the root, not inside it
    sibling.mkdir()
    (store.root / "notes.txt").write_text("not a session")
    (store.root / "plain-dir").mkdir()
    link = store.root / ("l" * 43)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pass  # symlinks need privileges on Windows; the other cases still run
    sid = store.create()
    assert store.cleanup_expired(now=clock.now + 10) == [sid]
    assert (outside / "keep.txt").read_text() == "keep"
    assert sibling.is_dir()
    assert (store.root / "notes.txt").exists() and (store.root / "plain-dir").is_dir()


def test_missing_marker_counts_as_expired(tmp_path: Path) -> None:
    store = make(tmp_path, Clock())
    sid = store.create()
    (store.root / sid / ".touched").unlink()
    assert store.cleanup_expired() == [sid]


def test_delete(tmp_path: Path) -> None:
    store = make(tmp_path, Clock())
    sid = store.create()
    store.delete(sid)
    assert not (store.root / sid).exists()


def test_a_sweep_during_create_cannot_delete_the_new_session(tmp_path: Path) -> None:
    stores: list[SessionStore] = []
    swept: list[list[str]] = []

    def clock() -> float:  # a concurrent sweep fires while the new session is being built
        if stores and not swept:
            swept.append(stores[0].cleanup_expired(now=1e12))
        return 1000.0

    store = SessionStore(tmp_path / "sessions", 100.0, clock)
    stores.append(store)
    sid = store.create()
    assert swept == [[]]
    assert store.path(sid).is_dir()


def test_a_directory_that_cannot_be_removed_does_not_stop_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    clock = Clock()
    store = make(tmp_path, clock)
    stuck, other = sorted([store.create(), store.create()])
    real = shutil.rmtree

    def rmtree(path: Path, *args: object, **kwargs: object) -> None:
        if Path(path).name == stuck:
            raise PermissionError("in use")
        real(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    clock.now += 1000
    assert store.cleanup_expired() == [other]
    assert (store.root / stuck).is_dir()
