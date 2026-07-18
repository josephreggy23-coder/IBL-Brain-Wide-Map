from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, Field


class RecoveryState(StrEnum):
    RESTITUTION = "restitution"
    COMPENSATION = "compensation"
    PERSISTENT_DYSFUNCTION = "persistent_dysfunction"
    UNCERTAIN = "uncertain"


class PoseObservation(BaseModel):
    dataset_id: str
    animal_id: str
    session_id: str
    video_id: str | None = None
    frame: int = Field(ge=0)
    time_seconds: float | None = None
    bodypart: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    likelihood: float | None = Field(default=None, ge=0, le=1)
    condition: str | None = None
    injury_status: str | None = None
    timepoint: str | None = None
    days_post_injury: float | None = None
    task: str | None = None
    sex: str | None = None
    speed: float | None = None
    camera_view: str | None = None


REQUIRED_FEATURE_COLUMNS = {"dataset_id", "animal_id", "session_id", "timepoint", "condition"}
