from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Protocol

from collections import defaultdict
import re

from asammdf import MDF  # pivotal dependency for MDF file handling
import numpy as np


@dataclass
class RawSegmentInfo:
    """
    Metadata + lazy loader for a single MDF segment (one channel in one measurement).

    Examples of measurement_name:
    - "RecResult[1]"
    - "D[3]"
    - "SomeOtherKey"
    """

    measurement_name: str      # "RecResult[1]", "D[3]", etc.
    key: str                   # "RecResult", "D", ...
    index: int | None          # 1, 2, 3... if present in [ ]
    source_path: str           # full source path in the MDF
    channel_name: str          # "eng_spd", "torque", ...
    unit: str | None
    n_samples: int
    t_start: float | None      # absolute or relative time (optional, can be None)
    t_end: float | None
    # MDF identifiers
    group_index: int           # group id inside the MDF
    channel_index: int         # channel id inside the group

    # Lazy loader: when called, reads ONLY this segment
    loader: Callable[[], tuple["np.ndarray", "np.ndarray"]]
    # -> (time, values)


@dataclass
class RawChannelInfo:
    """
    Logical channel view (potentially spanning multiple measurements).

    A logical channel is identified by (logical_name, key), e.g.:
    - logical_name = "eng_spd", key = "RecResult"  -> concatenation of RecResult[1..N]
    - logical_name = "eng_spd", key = "D"          -> concatenation of D[1..M]
    """

    logical_name: str           # e.g. "eng_spd"
    key: str                    # e.g. "RecResult" or "D"
    segments: list[RawSegmentInfo]
    unit: str | None
    dtype: "np.dtype | None"    # data type (float32, float64, ...)

    def load(self, with_measure_id: bool = False):
        """Load all segments and concatenate them.

        Parameters
        ----------
        with_measure_id:
            If True, also return an array of measurement indices (seg.index).

        Returns
        -------
        time, values
            If with_measure_id is False.
        time, values, meas_id
            If with_measure_id is True.
        """
        times: list[np.ndarray] = []
        values: list[np.ndarray] = []
        meas_ids: list[np.ndarray] = []

        for seg in self.segments:
            t, v = seg.loader()
            times.append(t)
            values.append(v)
            if with_measure_id:
                idx = seg.index if seg.index is not None else -1
                meas_ids.append(
                    np.full(t.shape, fill_value=idx, dtype=np.int32)
                )

        if not times:
            if with_measure_id:
                empty = np.array([])
                return empty, empty, empty.astype(np.int32)
            empty = np.array([])
            return empty, empty

        time_array = np.concatenate(times)
        value_array = np.concatenate(values)

        if with_measure_id:
            meas_id_array = np.concatenate(meas_ids)
            return time_array, value_array, meas_id_array
        return time_array, value_array


class MdfReader(Protocol):
    """Protocol for MDF readers.

    Implementations should expose a logical view of channels.
    """

    def list_channels(self) -> List[RawChannelInfo]:
        ...

    def read_channels(
        self,
        channel_names: Iterable[str],
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, "RawChannelData"]:
        ...


@dataclass
class RawChannelData:
    time: "np.ndarray"
    values: "np.ndarray"


_MEAS_RE = re.compile(r"^(?P<key>[A-Za-z]+)(?:\[(?P<idx>\d+)\])?$")


def _parse_key_and_index(meas_name: str) -> tuple[str, int | None]:
    """Parse measurement name into (key, index).

    Examples
    --------
    "RecResult[1]" -> ("RecResult", 1)
    "D[3]"         -> ("D", 3)
    "Foo"          -> ("Foo", None)
    """
    m = _MEAS_RE.match(meas_name)
    if not m:
        # Fallback: treat the whole name as key, no index
        return meas_name, None
    key = m.group("key")
    idx_str = m.group("idx")
    return key, int(idx_str) if idx_str is not None else None


def _extract_measurement_name(source_path: str) -> str:
    """Extract measurement name from source_path.

    This is heuristic and may be refined once real MDF files are inspected.
    Typically, Concerto encodes the measurement as the last component
    of a path like ".../RecResult[1]".
    """
    if not source_path:
        return ""
    parts = source_path.replace("\\", "/").split("/")
    return parts[-1]


class AsammdfReader:
    """Concrete implementation of MdfReader using asammdf.MDF.

    It groups channels by (key, logical_name), where `key` is typically
    something like "RecResult" or "D", extracted from the source path
    of the MDF channel.
    """

    # By default, when the protocol-level API asks for a channel only
    # by name (without key), we use this key as the primary view.
    _DEFAULT_KEY = "RecResult"

    def __init__(self, path: str):
        self._mdf = MDF(path)
        # (key, logical_name) -> RawChannelInfo
        self._channels_by_key_name: dict[tuple[str, str], RawChannelInfo] = {}
        # logical_name -> default RawChannelInfo (prefer DEFAULT_KEY if available)
        self._logical_index: dict[str, RawChannelInfo] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        temp_segments: dict[tuple[str, str], list[RawSegmentInfo]] = defaultdict(list)

        for group_index, group in enumerate(self._mdf.groups):
            for channel_index, channel in enumerate(group.channels):
                # `source_path` naming convention is MDF/Concerto-specific;
                # we try a few attribute names and fall back to "".
                source_path = (
                    getattr(channel, "source_path", None)
                    or getattr(channel, "source", None)
                    or ""
                )
                meas_name = _extract_measurement_name(source_path)
                key, idx = _parse_key_and_index(meas_name)

                def make_loader(g_i: int = group_index, c_i: int = channel_index):
                    def _loader() -> tuple[np.ndarray, np.ndarray]:
                        sig = self._mdf.get(group=g_i, index=c_i)
                        # asammdf Signal interface: timestamps & samples
                        t = sig.timestamps
                        v = sig.samples
                        return t, v

                    return _loader

                # t_start / t_end left as None for now (lazy evaluation possible later)
                seg = RawSegmentInfo(
                    measurement_name=meas_name,
                    key=key,
                    index=idx,
                    source_path=source_path,
                    channel_name=channel.name,
                    unit=channel.unit,
                    n_samples=channel.samples_count,
                    t_start=None,
                    t_end=None,
                    group_index=group_index,
                    channel_index=channel_index,
                    loader=make_loader(),
                )

                temp_segments[(key, channel.name)].append(seg)

        # Build RawChannelInfo objects from collected segments
        for (key, ch_name), segments in temp_segments.items():
            # Order segments by index (RecResult[1], [2], ...)
            ordered = sorted(
                segments,
                key=lambda s: (s.index if s.index is not None else -1),
            )
            unit = ordered[0].unit if ordered else None

            ch_info = RawChannelInfo(
                logical_name=ch_name,
                key=key,
                segments=ordered,
                unit=unit,
                dtype=None,
            )
            self._channels_by_key_name[(key, ch_name)] = ch_info

        # Build logical index: for each logical name, prefer DEFAULT_KEY
        by_name: dict[str, list[RawChannelInfo]] = defaultdict(list)
        for (key, ch_name), ch_info in self._channels_by_key_name.items():
            by_name[ch_name].append(ch_info)

        for name, infos in by_name.items():
            # Try to pick the DEFAULT_KEY view if available
            default = None
            for info in infos:
                if info.key == self._DEFAULT_KEY:
                    default = info
                    break
            if default is None:
                # Fallback: take the first available key
                default = infos[0]
            self._logical_index[name] = default

    # ------------------------------------------------------------------
    # MdfReader protocol implementation
    # ------------------------------------------------------------------
    def list_channels(self) -> List[RawChannelInfo]:
        """List logical channels (one entry per logical_name).

        By default, for channels with multiple keys (e.g. RecResult, D),
        this returns the view associated with _DEFAULT_KEY (RecResult) if
        available, otherwise the first available key.
        """
        return list(self._logical_index.values())

    def read_channels(
        self,
        channel_names: Iterable[str],
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, RawChannelData]:
        """Read multiple channels by logical name.

        Parameters
        ----------
        channel_names:
            Iterable of logical channel names (e.g. "eng_spd"). The key
            used is the default one (RecResult) if available.
        start_time, end_time:
            Optional time window for slicing. Time base is assumed to be
            the one returned by asammdf (typically seconds).
        """
        result: dict[str, RawChannelData] = {}

        for name in channel_names:
            if name not in self._logical_index:
                raise KeyError(f"Channel '{name}' not found in MDF (by logical name)")

            ch_info = self._logical_index[name]
            t, v = ch_info.load(with_measure_id=False)

            # Apply optional time window
            if start_time is not None or end_time is not None:
                mask = np.ones_like(t, dtype=bool)
                if start_time is not None:
                    mask &= t >= start_time
                if end_time is not None:
                    mask &= t <= end_time
                t = t[mask]
                v = v[mask]

            result[name] = RawChannelData(time=t, values=v)

        return result