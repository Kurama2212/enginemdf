# enginemdf/core/metadata.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exceptions import InvalidChannel, InvalidSegment, InvalidDataset


@dataclass(frozen=True, slots=True)
class ChannelMeta:
    """
    Metadata attached to a Channel.

    Keep it lightweight and extensible:
    - unit: display/physical unit (rpm, Nm, ...)
    - description: human-friendly description
    - source: origin (ECU, sensor, computed, ...)
    - attrs: arbitrary additional fields
    """
    unit: str | None = None
    description: str | None = None
    source: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        elif not isinstance(self.attrs, dict):
            raise InvalidChannel("ChannelMeta.attrs must be a dict.")


@dataclass(frozen=True, slots=True)
class SegmentMeta:
    """
    Metadata attached to a Segment (a run/measurement).

    Example kinds:
    - power_curve
    - calibration
    - track_sim
    """
    kind: str | None = None
    description: str | None = None
    source: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        elif not isinstance(self.attrs, dict):
            raise InvalidSegment("SegmentMeta.attrs must be a dict.")


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    """
    Metadata attached to a Dataset (collection of segments).
    """
    description: str | None = None
    source: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        elif not isinstance(self.attrs, dict):
            raise InvalidDataset("DatasetMeta.attrs must be a dict.")