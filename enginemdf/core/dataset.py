# enginemdf/core/dataset.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

from .metadata import DatasetMeta
from .exceptions import InvalidDataset, SegmentNotFound
from .segment import Segment


class DatasetError(Exception):
    """Base error for Dataset-related failures."""



@dataclass(frozen=True, slots=True)
class Dataset:
    """
    Dataset = collection of Segments.

    Design goals:
    - dict-like access: ds["PowerCurve_01"]
    - safe + predictable: immutable, validated, consistent naming
    - composable transformations: add/drop/select/slice_time return new Dataset
    """
    segments: Mapping[str, Segment] = field(default_factory=dict, repr=False)
    meta: DatasetMeta = field(default_factory=DatasetMeta, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.segments, Mapping):
            raise InvalidDataset("Dataset.segments must be a mapping (e.g., dict).")
        if not isinstance(self.meta, DatasetMeta):
            raise InvalidDataset("Dataset.meta must be a DatasetMeta instance.")

        normalized: dict[str, Segment] = {}
        for key, seg in self.segments.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidDataset("Dataset.segments keys must be non-empty strings.")
            if not isinstance(seg, Segment):
                raise InvalidDataset("Dataset.segments values must be Segment instances.")
            if seg.name != key:
                raise InvalidDataset(
                    f"Segment name mismatch: key '{key}' but Segment.name is '{seg.name}'."
                )
            normalized[key] = seg

        object.__setattr__(self, "segments", normalized)

    # ---- dict-like API ----
    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[str]:
        return iter(self.segments)

    def __contains__(self, name: object) -> bool:
        return name in self.segments

    def keys(self) -> Iterable[str]:
        return self.segments.keys()

    def items(self) -> Iterable[tuple[str, Segment]]:
        return self.segments.items()

    def values(self) -> Iterable[Segment]:
        return self.segments.values()

    def __getitem__(self, name: str) -> Segment:
        try:
            return self.segments[name]
        except KeyError as e:
            raise SegmentNotFound(name) from e

    def get(self, name: str, default: Segment | None = None) -> Segment | None:
        return self.segments.get(name, default)

    # ---- derived time bounds ----
    @property
    def t_start(self) -> float | None:
        starts = [seg.t_start for seg in self.segments.values() if seg.t_start is not None]
        return None if not starts else float(min(starts))

    @property
    def t_end(self) -> float | None:
        ends = [seg.t_end for seg in self.segments.values() if seg.t_end is not None]
        return None if not ends else float(max(ends))

    # ---- transformations ----
    def add(self, segment: Segment, *, overwrite: bool = False) -> "Dataset":
        """
        Return a new Dataset with `segment` added.

        If overwrite=False and the segment already exists, raises InvalidDataset.
        """
        if not isinstance(segment, Segment):
            raise InvalidDataset("add() expects a Segment instance.")

        name = segment.name
        if (name in self.segments) and not overwrite:
            raise InvalidDataset(f"Segment '{name}' already exists (overwrite=False).")

        new_segments = dict(self.segments)
        new_segments[name] = segment
        return Dataset(segments=new_segments, meta=self._copy_meta())

    def drop(self, names: str | Iterable[str], *, missing: str = "raise") -> "Dataset":
        """
        Drop one or more segments.

        missing:
          - "raise": error if any name is missing
          - "ignore": skip missing names
        """
        if isinstance(names, str):
            names_set = {names}
        else:
            names_set = set(names)

        new_segments = dict(self.segments)
        for n in names_set:
            if n in new_segments:
                del new_segments[n]
            elif missing == "raise":
                raise SegmentNotFound(n)
        return Dataset(segments=new_segments, meta=self._copy_meta())

    def select(self, names: Iterable[str], *, missing: str = "raise") -> "Dataset":
        """
        Keep only the given segment names (order preserved by insertion in `names`).

        missing:
          - "raise": error if any name is missing
          - "ignore": skip missing names
        """
        selected: dict[str, Segment] = {}
        for n in names:
            if n in self.segments:
                selected[n] = self.segments[n]
            elif missing == "raise":
                raise SegmentNotFound(n)
        return Dataset(segments=selected, meta=self._copy_meta())

    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
        drop_empty_segments: bool = False,
        drop_empty_channels: bool = False,
    ) -> "Dataset":
        """
        Slice all segments by time.

        drop_empty_segments:
          - False: keep segments even if they become empty (0 channels)
          - True : remove segments that have 0 channels after slicing

        drop_empty_channels:
          - passed to Segment.slice_time(drop_empty=...)
        """
        new_segments: dict[str, Segment] = {}
        for name, seg in self.segments.items():
            seg_sliced = seg.slice_time(t_min, t_max, closed=closed, drop_empty=drop_empty_channels)
            if drop_empty_segments and len(seg_sliced) == 0:
                continue
            new_segments[name] = seg_sliced

        return Dataset(segments=new_segments, meta=self._copy_meta())

    def rename_segment(self, old: str, new: str) -> "Dataset":
        """
        Rename a segment key and its Segment.name consistently.
        """
        if old not in self.segments:
            raise SegmentNotFound(old)
        if not isinstance(new, str) or not new.strip():
            raise InvalidDataset("New segment name must be a non-empty string.")
        if new in self.segments:
            raise InvalidDataset(f"Segment '{new}' already exists.")

        new_segments = dict(self.segments)
        seg = new_segments.pop(old)
        new_segments[new] = seg.rename(new)
        return Dataset(segments=new_segments, meta=self._copy_meta())

    def merge(self, other: "Dataset", *, overwrite: bool = False) -> "Dataset":
        """
        Merge two datasets (by segment name).

        If overwrite=False, raises if a segment name collides.
        """
        if not isinstance(other, Dataset):
            raise InvalidDataset("merge() expects a Dataset instance.")

        new_segments = dict(self.segments)
        for name, seg in other.segments.items():
            if (name in new_segments) and not overwrite:
                raise InvalidDataset(f"Segment '{name}' collides (overwrite=False).")
            new_segments[name] = seg

        # Keep current meta; you can extend this later (merge meta rules)
        return Dataset(segments=new_segments, meta=self._copy_meta())

    def _copy_meta(self) -> DatasetMeta:
        return DatasetMeta(
            description=self.meta.description,
            source=self.meta.source,
            attrs=self.meta.attrs.copy(),
        )