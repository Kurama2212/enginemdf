# core/channel.py

from __future__ import annotations

from dataclasses import dataclass, field

from .timeseries import TimeSeriesLike
from .exceptions import InvalidChannel
from .metadata import ChannelMeta

import numpy as np


class ChannelError(Exception):
    """Base error for Channel-related failures."""


@dataclass(slots=True, frozen=True)
class Channel:
    name: str
    series: TimeSeriesLike
    meta: ChannelMeta = field(default_factory=ChannelMeta)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidChannel("Channel.name must be a non-empty string.")

        if not isinstance(self.series, TimeSeriesLike):
            raise InvalidChannel("Channel.series must be a TimeSeries-like object.")

        if not isinstance(self.meta, ChannelMeta):
            raise InvalidChannel("Channel.meta must be a ChannelMeta instance.")

        # If meta.unit is not provided, inherit from series.unit (if any).
        if self.meta.unit is None and self.series.unit is not None:
            object.__setattr__(
                self,
                "meta",
                ChannelMeta(
                    unit=self.series.unit,
                    description=self.meta.description,
                    source=self.meta.source,
                    attrs=self.meta.attrs.copy(),
                ),
            )

    # Convenience accessors
    @property
    def time(self) -> np.ndarray:
        return self.series.time

    @property
    def values(self) -> np.ndarray:
        return self.series.values

    @property
    def unit(self) -> str | None:
        # Meta takes precedence
        return self.meta.unit if self.meta.unit is not None else self.series.unit

    @property
    def n(self) -> int:
        return self.series.n

    @property
    def t_start(self) -> float | None:
        return self.series.t_start

    @property
    def t_end(self) -> float | None:
        return self.series.t_end

    # Core operations
    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
    ) -> "Channel":
        return Channel(
            name=self.name,
            series=self.series.slice_time(t_min, t_max, closed=closed),
            meta=ChannelMeta(
                unit=self.meta.unit,
                description=self.meta.description,
                source=self.meta.source,
                attrs=self.meta.attrs.copy(),
            ),
        )

    def rename(self, name: str) -> "Channel":
        return Channel(name=name, series=self.series, meta=self._copy_meta())

    def with_unit(self, unit: str | None) -> "Channel":
        # Keep series as-is; override unit at meta level
        return Channel(
            name=self.name,
            series=self.series,
            meta=ChannelMeta(
                unit=unit,
                description=self.meta.description,
                source=self.meta.source,
                attrs=self.meta.attrs.copy(),
            ),
        )

    def with_description(self, description: str | None) -> "Channel":
        return Channel(
            name=self.name,
            series=self.series,
            meta=ChannelMeta(
                unit=self.meta.unit,
                description=description,
                source=self.meta.source,
                attrs=self.meta.attrs.copy(),
            ),
        )

    def with_source(self, source: str | None) -> "Channel":
        return Channel(
            name=self.name,
            series=self.series,
            meta=ChannelMeta(
                unit=self.meta.unit,
                description=self.meta.description,
                source=source,
                attrs=self.meta.attrs.copy(),
            ),
        )

    def to_numpy(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        return self.series.to_numpy(copy=copy)

    def _copy_meta(self) -> ChannelMeta:
        return ChannelMeta(
            unit=self.meta.unit,
            description=self.meta.description,
            source=self.meta.source,
            attrs=self.meta.attrs.copy(),
        )