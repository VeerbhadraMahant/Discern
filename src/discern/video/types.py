"""Video data model (system-design 6): probe results, sampled frames, shots and tracks."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from discern.models.roles import Image
from discern.vision.boxes import Box, scale


class VideoError(ValueError):
    """The input cannot be decoded or violates a profile limit."""


class VideoInfo(BaseModel):
    fps: float
    duration: float  # seconds
    width: int
    height: int
    frame_count: int


class SampledFrame(BaseModel):
    """One sampled frame at working resolution, with what is needed to map back."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int  # decode-order frame number in the original video
    time: float  # seconds
    image: Image  # RGB uint8, long side at most max_long_side_px
    original_size: tuple[int, int]  # (width, height) of the original frame

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    @property
    def scale_factor(self) -> float:
        """Working-resolution width over original width (1.0 when not downscaled)."""
        return self.size[0] / self.original_size[0]

    def to_original(self, box: Box) -> Box:
        """Map a working-resolution box into the original frame's coordinates."""
        sx, sy = self.original_size[0] / self.size[0], self.original_size[1] / self.size[1]
        return scale(box, sx, sy)


class Shot(BaseModel):
    id: int
    t_start: float
    t_end: float
    keyframe_index: int
    keyframe_time: float


class TrackCrop(BaseModel):
    """One of a track's best crops, ranked by box area x score x sharpness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int
    time: float
    box: Box  # original-frame coordinates
    rank: float
    image: Image


class Track(BaseModel):
    """Instance groups linked across sampled frames. Boxes are in original-frame pixels."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    frames: dict[int, Box]  # sampled frame index -> box
    timestamps: dict[int, float]  # sampled frame index -> seconds
    scores: dict[int, float]  # sampled frame index -> detection score
    label: str  # majority detector label until adjudication, then the adjudicated label
    best_crops: list[TrackCrop] = Field(default_factory=list)  # best first
    status: Literal["accepted", "rejected"] = "accepted"  # set by adjudication
    rationale: str = ""

    @property
    def t_start(self) -> float:
        return min(self.timestamps.values())

    @property
    def t_end(self) -> float:
        return max(self.timestamps.values())

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores)
