"""Feedback record and its private JSONL store (system-design 5.7, 11).

Without `retain_media` only labels and box coordinates are kept: no image or video bytes are
copied into the store. With it, the media file is copied under a generated name.
"""

import secrets
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from discern.eval.types import AnnotatedImage, GroundTruthBox
from discern.serve.session import ALLOWED_EXTENSIONS
from discern.vision.boxes import Box

Verdict = Literal["correct", "wrong", "missing"]
_IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})


class Feedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    result_set_id: str
    verdict: Verdict
    track_id: int | None = None
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 in original pixels
    label: str = ""  # object label the verdict is about
    note: str = ""
    retain_media: bool = False  # explicit opt-in to keep the media for training
    image_size: tuple[int, int] | None = None  # (width, height) of the media, if known
    session_ref: str = ""  # hash of the session id: result set ids ("R1") repeat across sessions

    @model_validator(mode="after")
    def _missing_needs_a_box(self) -> Self:
        if self.verdict == "missing" and self.box is None:
            raise ValueError("a missing-object mark needs the user's box")
        return self


class StoredFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    feedback: Feedback
    media_file: str | None = None  # name inside the store's media directory; opt-in only


class LabelledSample(BaseModel):
    """A harvest-ready image plus boxes the user rejected (hard negatives)."""

    model_config = ConfigDict(frozen=True)

    image: AnnotatedImage
    negatives: tuple[GroundTruthBox, ...] = ()


class FeedbackStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "feedback.jsonl"
        self.media_dir = root / "media"

    def add(self, feedback: Feedback, media: Path | None = None) -> StoredFeedback:
        """Append feedback. `media` is copied in only when `feedback.retain_media` is True."""
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)  # private to the owner where the platform supports it
        except OSError:
            pass
        item_id = secrets.token_hex(8)
        media_file: str | None = None
        if feedback.retain_media and media is not None:
            extension = media.suffix.lstrip(".").lower()
            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError(f"media type {extension!r} is not allowed")
            self.media_dir.mkdir(exist_ok=True)
            media_file = f"{item_id}.{extension}"
            shutil.copyfile(media, self.media_dir / media_file)
        stored = StoredFeedback(id=item_id, feedback=feedback, media_file=media_file)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(stored.model_dump_json() + "\n")
        return stored

    def load(self) -> list[StoredFeedback]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [StoredFeedback.model_validate_json(line) for line in f if line.strip()]


def _usable(item: StoredFeedback) -> bool:
    fb = item.feedback
    return bool(
        fb.retain_media
        and item.media_file is not None
        and item.media_file.rsplit(".", 1)[-1] in _IMAGE_EXTENSIONS
        and fb.image_size is not None
        and fb.box is not None
        and fb.label
    )


def feedback_to_labelled_samples(
    items: Sequence[StoredFeedback], store: FeedbackStore
) -> list[LabelledSample]:
    """One sample per result set. Only opt-in feedback with retained image media, a known image
    size and a labelled box contributes; everything else is skipped. The first usable item's
    media file stands for the result set.

    `correct` boxes and `missing` boxes (the user's own) become ground truth objects; `wrong`
    boxes become negatives.
    """
    groups: dict[tuple[str, str], list[StoredFeedback]] = {}
    for item in items:
        if _usable(item):
            fb = item.feedback
            groups.setdefault((fb.session_ref, fb.result_set_id), []).append(item)
    samples: list[LabelledSample] = []
    for (session_ref, result_set_id), members in sorted(groups.items()):
        media_file = members[0].media_file or ""
        objects: list[GroundTruthBox] = []
        negatives: list[GroundTruthBox] = []
        width, height = members[0].feedback.image_size or (0, 0)
        for m in members:
            if m.feedback.box is None:
                continue
            truth = GroundTruthBox(box=Box(*m.feedback.box), label=m.feedback.label)
            (negatives if m.feedback.verdict == "wrong" else objects).append(truth)
        image = AnnotatedImage(
            image_id="-".join(("feedback", *filter(None, (session_ref, result_set_id)))),
            path=store.media_dir / media_file,
            width=width,
            height=height,
            objects=tuple(objects),
            split="harvest",
        )
        samples.append(LabelledSample(image=image, negatives=tuple(negatives)))
    return samples
