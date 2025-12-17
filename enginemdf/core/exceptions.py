# enginemdf/core/exceptions.py
from __future__ import annotations


class CoreError(Exception):
    """Base error for all core-domain exceptions."""


# ---- Validation / construction errors ----
class InvalidTimeSeries(CoreError):
    """Raised when a TimeSeries is constructed with invalid inputs."""


class InvalidChannel(CoreError):
    """Raised when a Channel / ChannelMeta is constructed with invalid inputs."""


class InvalidSegment(CoreError):
    """Raised when a Segment / SegmentMeta is constructed with invalid inputs."""


class InvalidDataset(CoreError):
    """Raised when a Dataset / DatasetMeta is constructed with invalid inputs."""


# ---- Lookup errors (also behave like KeyError for dict-like APIs) ----
class ChannelNotFound(CoreError, KeyError):
    """Raised when a requested channel name is not present."""


class SegmentNotFound(CoreError, KeyError):
    """Raised when a requested segment name is not present."""


