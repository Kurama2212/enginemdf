# enginemdf/core/__init__.py
"""
Core domain objects for enginemdf.

This module defines the high-level, MDF-agnostic data model:
- TimeSeries: validated 1D signal over time
- Channel: logical signal (name + TimeSeries + metadata)
- Segment: coherent measurement/run containing multiple channels
- Dataset: collection of segments

The core layer is independent from I/O and storage formats.
"""

from .timeseries import TimeSeries, LazyTimeSeries, TimeSeriesLike
from .channel import Channel
from .segment import Segment
from .dataset import Dataset
from .metadata import ChannelMeta, SegmentMeta, DatasetMeta
from .exceptions import (
    CoreError,
    InvalidTimeSeries,
    InvalidChannel,
    InvalidSegment,
    InvalidDataset,
    ChannelNotFound,
    SegmentNotFound,
)


__all__ = [
    # time series
    "TimeSeries",
    "LazyTimeSeries",
    "TimeSeriesLike",

    # domain objects
    "Channel",
    "Segment",
    "Dataset",

    # metadata
    "ChannelMeta",
    "SegmentMeta",
    "DatasetMeta",

    # exceptions
    "CoreError",
    "InvalidTimeSeries",
    "InvalidChannel",
    "InvalidSegment",
    "InvalidDataset",
    "ChannelNotFound",
    "SegmentNotFound",
]