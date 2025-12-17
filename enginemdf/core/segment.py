# enginemdf/core/segment.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

from .exceptions import ChannelNotFound, InvalidSegment
from .metadata import SegmentMeta
from .channel import Channel


class SegmentError(Exception):
    """Base error for Segment-related failures."""




@dataclass(frozen=True, slots=True)
class Segment:
    """
    A Segment represents a coherent measurement/run containing multiple Channels.

    Design goals:
    - easy access: segment["eng_spd"]
    - safe: validate channel container and names
    - predictable: immutable; transformations return new Segment
    """
    name: str
    channels: Mapping[str, Channel] = field(default_factory=dict, repr=False)
    meta: SegmentMeta = field(default_factory=SegmentMeta, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidSegment("Segment.name must be a non-empty string.")
        if not isinstance(self.channels, Mapping):
            raise InvalidSegment("Segment.channels must be a mapping (e.g., dict).")
        if not isinstance(self.meta, SegmentMeta):
            raise InvalidSegment("Segment.meta must be a SegmentMeta instance.")

        # Validate channel mapping consistency
        normalized: dict[str, Channel] = {}
        for key, ch in self.channels.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidSegment("Segment.channels keys must be non-empty strings.")
            if not isinstance(ch, Channel):
                raise InvalidSegment("Segment.channels values must be Channel instances.")
            # Enforce key-name consistency (important for predictable API)
            if ch.name != key:
                raise InvalidSegment(
                    f"Channel name mismatch: key '{key}' but Channel.name is '{ch.name}'."
                )
            normalized[key] = ch

        # Freeze channels to a plain dict (stable / predictable)
        object.__setattr__(self, "channels", normalized)

    # ---- dict-like API ----
    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[str]:
        return iter(self.channels)

    def keys(self) -> Iterable[str]:
        return self.channels.keys()

    def items(self) -> Iterable[tuple[str, Channel]]:
        return self.channels.items()

    def values(self) -> Iterable[Channel]:
        return self.channels.values()

    def __contains__(self, name: object) -> bool:
        return name in self.channels

    def __getitem__(self, name: str) -> Channel:
        try:
            return self.channels[name]
        except KeyError as e:
            raise ChannelNotFound(name) from e

    def get(self, name: str, default: Channel | None = None) -> Channel | None:
        return self.channels.get(name, default)

    # ---- derived time bounds ----
    @property
    def t_start(self) -> float | None:
        starts = [ch.t_start for ch in self.channels.values() if ch.t_start is not None]
        return None if not starts else float(min(starts))

    @property
    def t_end(self) -> float | None:
        ends = [ch.t_end for ch in self.channels.values() if ch.t_end is not None]
        return None if not ends else float(max(ends))

    # ---- transformations ----
    def add(self, channel: Channel, *, overwrite: bool = False) -> "Segment":
        """
        Return a new Segment with `channel` added.

        If overwrite=False and the channel already exists, raises InvalidSegment.
        """
        if not isinstance(channel, Channel):
            raise InvalidSegment("add() expects a Channel instance.")

        name = channel.name
        if (name in self.channels) and not overwrite:
            raise InvalidSegment(f"Channel '{name}' already exists (overwrite=False).")

        new_channels = dict(self.channels)
        new_channels[name] = channel
        return Segment(name=self.name, channels=new_channels, meta=self._copy_meta())

    def drop(self, names: str | Iterable[str], *, missing: str = "raise") -> "Segment":
        """
        Drop one or more channels.

        missing:
          - "raise": error if any name is missing
          - "ignore": skip missing names
        """
        if isinstance(names, str):
            names_set = {names}
        else:
            names_set = set(names)

        new_channels = dict(self.channels)
        for n in names_set:
            if n in new_channels:
                del new_channels[n]
            elif missing == "raise":
                raise ChannelNotFound(n)
        return Segment(name=self.name, channels=new_channels, meta=self._copy_meta())

    def select(self, names: Iterable[str], *, missing: str = "raise") -> "Segment":
        """
        Keep only the given channel names (order preserved by insertion in `names`).

        missing:
          - "raise": error if any name is missing
          - "ignore": skip missing names
        """
        selected: dict[str, Channel] = {}
        for n in names:
            if n in self.channels:
                selected[n] = self.channels[n]
            elif missing == "raise":
                raise ChannelNotFound(n)
        return Segment(name=self.name, channels=selected, meta=self._copy_meta())

    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
        drop_empty: bool = False,
    ) -> "Segment":
        """
        Slice all channels by time.

        drop_empty:
          - False: keep channels even if slicing results in empty series
          - True : remove channels that become empty after slicing
        """
        new_channels: dict[str, Channel] = {}
        for name, ch in self.channels.items():
            ch_sliced = ch.slice_time(t_min, t_max, closed=closed)
            if drop_empty and ch_sliced.n == 0:
                continue
            new_channels[name] = ch_sliced

        return Segment(name=self.name, channels=new_channels, meta=self._copy_meta())

    def rename(self, name: str) -> "Segment":
        return Segment(name=name, channels=dict(self.channels), meta=self._copy_meta())

    def _copy_meta(self) -> SegmentMeta:
        return SegmentMeta(
            kind=self.meta.kind,
            description=self.meta.description,
            source=self.meta.source,
            attrs=self.meta.attrs.copy(),
        )