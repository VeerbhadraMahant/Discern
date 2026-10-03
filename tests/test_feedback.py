from pathlib import Path

import pytest
from pydantic import ValidationError

from discern.feedback.schema import (
    Feedback,
    FeedbackStore,
    feedback_to_labelled_samples,
)

BOX = (10.0, 20.0, 110.0, 220.0)
MEDIA_BYTES = b"\x89PNG-secret-pixels-0123456789"


def media(tmp_path: Path) -> Path:
    path = tmp_path / "upload.png"
    path.write_bytes(MEDIA_BYTES)
    return path


def all_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in root.rglob("*") if p.is_file())


def fb(verdict: str, box: tuple[float, float, float, float] | None, label: str = "car") -> Feedback:
    return Feedback.model_validate(
        {
            "result_set_id": "R1",
            "verdict": verdict,
            "box": box,
            "label": label,
            "retain_media": True,
            "image_size": (640, 480),
        }
    )


def test_no_media_retained_without_opt_in(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb")
    feedback = Feedback(result_set_id="R1", verdict="correct", box=BOX, label="car")
    assert feedback.retain_media is False
    stored = store.add(feedback, media=media(tmp_path))
    assert stored.media_file is None
    assert not store.media_dir.exists()
    assert b"secret-pixels" not in all_bytes(store.root)
    assert "110.0" in store.path.read_text(encoding="utf-8")  # box coordinates are kept
    assert store.load() == [stored]


def test_media_copied_with_opt_in(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb")
    stored = store.add(fb("correct", BOX), media=media(tmp_path))
    assert stored.media_file is not None and stored.media_file.endswith(".png")
    assert (store.media_dir / stored.media_file).read_bytes() == MEDIA_BYTES


def test_missing_requires_a_box() -> None:
    with pytest.raises(ValidationError):
        Feedback(result_set_id="R1", verdict="missing", label="car")


def test_bad_media_type_rejected(tmp_path: Path) -> None:
    exe = tmp_path / "x.exe"
    exe.write_bytes(b"x")
    with pytest.raises(ValueError):
        FeedbackStore(tmp_path / "fb").add(fb("wrong", BOX), media=exe)


def test_conversion_to_samples(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb")
    src = media(tmp_path)
    items = [
        store.add(fb("correct", BOX), media=src),
        store.add(fb("wrong", (0, 0, 5, 5)), media=src),
        store.add(fb("missing", (50, 50, 90, 90), "bus"), media=src),
    ]
    samples = feedback_to_labelled_samples(items, store)
    assert len(samples) == 1
    s = samples[0]
    assert s.image.width == 640 and s.image.height == 480 and s.image.split == "harvest"
    assert [(o.label, tuple(o.box)) for o in s.image.objects] == [
        ("car", BOX),
        ("bus", (50.0, 50.0, 90.0, 90.0)),
    ]
    assert [tuple(n.box) for n in s.negatives] == [(0.0, 0.0, 5.0, 5.0)]
    assert s.image.path == store.media_dir / (items[0].media_file or "")
    assert s.image.path.read_bytes() == MEDIA_BYTES


def test_conversion_skips_non_consented_and_unusable(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb")
    src = media(tmp_path)
    no_consent = store.add(
        Feedback(result_set_id="R1", verdict="correct", box=BOX, label="car", image_size=(9, 9)),
        media=src,
    )
    no_box = store.add(fb("wrong", None), media=src)
    no_label = store.add(fb("correct", BOX, ""), media=src)
    assert feedback_to_labelled_samples([no_consent, no_box, no_label], store) == []


def test_same_result_set_id_in_two_sessions_gives_two_samples(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb")
    src = media(tmp_path)
    items = [
        store.add(fb("correct", BOX).model_copy(update={"session_ref": ref}), media=src)
        for ref in ("aaaa", "bbbb")
    ]
    samples = feedback_to_labelled_samples(items, store)
    assert len(samples) == 2
    assert len({s.image.image_id for s in samples}) == 2
    assert all(len(s.image.objects) == 1 for s in samples)
