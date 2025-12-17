# core/timeseries.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


class TimeSeriesError(Exception):
    """Base error for TimeSeries-related failures."""


class InvalidTimeSeries(TimeSeriesError):
    """Raised when inputs are invalid (shape, dtype, ordering, etc.)."""


@runtime_checkable
class TimeSeriesLike(Protocol):
    """Structural interface implemented by both eager and lazy time series."""

    @property
    def time(self) -> np.ndarray: ...

    @property
    def values(self) -> np.ndarray: ...

    unit: str | None
    name: str | None
    attrs: dict[str, Any]

    @property
    def n(self) -> int: ...

    @property
    def t_start(self) -> float | None: ...

    @property
    def t_end(self) -> float | None: ...

    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
    ) -> "TimeSeriesLike": ...

    def to_numpy(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """Immutable (eager) time series: 1D time vector + 1D values vector."""

    time: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    unit: str | None = None
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        t = np.asarray(self.time)
        v = np.asarray(self.values)

        if t.ndim != 1:
            raise InvalidTimeSeries(f"`time` must be 1D, got shape {t.shape}")
        if v.ndim != 1:
            raise InvalidTimeSeries(f"`values` must be 1D, got shape {v.shape}")
        if t.size != v.size:
            raise InvalidTimeSeries(
                f"`time` and `values` must have same length, got {t.size} vs {v.size}"
            )

        if t.size > 0:
            if not np.isfinite(t).all():
                raise InvalidTimeSeries("`time` contains non-finite values (NaN/Inf).")
            dt = np.diff(t)
            if np.any(dt < 0):
                raise InvalidTimeSeries("`time` must be monotonic non-decreasing.")

        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        elif not isinstance(self.attrs, dict):
            raise InvalidTimeSeries("`attrs` must be a dict.")

        object.__setattr__(self, "time", t)
        object.__setattr__(self, "values", v)

    @property
    def n(self) -> int:
        return int(self.time.size)

    @property
    def t_start(self) -> float | None:
        return None if self.n == 0 else float(self.time[0])

    @property
    def t_end(self) -> float | None:
        return None if self.n == 0 else float(self.time[-1])

    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
    ) -> "TimeSeries":
        if closed not in {"both", "left", "right", "neither"}:
            raise ValueError("closed must be one of: both, left, right, neither")

        if self.n == 0:
            return self

        t = self.time
        mask = np.ones_like(t, dtype=bool)

        if t_min is not None:
            if closed in {"both", "left"}:
                mask &= (t >= t_min)
            else:
                mask &= (t > t_min)

        if t_max is not None:
            if closed in {"both", "right"}:
                mask &= (t <= t_max)
            else:
                mask &= (t < t_max)

        return TimeSeries(
            time=t[mask],
            values=self.values[mask],
            unit=self.unit,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def mean(self, *, skipna: bool = True) -> float | None:
        if self.n == 0:
            return None
        v = self.values
        if skipna and np.issubdtype(v.dtype, np.floating):
            return float(np.nanmean(v))
        return float(np.mean(v))

    def std(self, *, ddof: int = 0, skipna: bool = True) -> float | None:
        if self.n == 0:
            return None
        v = self.values
        if skipna and np.issubdtype(v.dtype, np.floating):
            return float(np.nanstd(v, ddof=ddof))
        return float(np.std(v, ddof=ddof))

    def to_numpy(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if copy:
            return self.time.copy(), self.values.copy()
        return self.time, self.values


@dataclass(slots=True)
class LazyTimeSeries:
    """Lazy time series: loads arrays on first access and caches them."""

    loader: Callable[[], tuple[np.ndarray, np.ndarray]] = field(repr=False)
    unit: str | None = None
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, repr=False)

    _time: np.ndarray | None = field(default=None, init=False, repr=False)
    _values: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.loader):
            raise InvalidTimeSeries("LazyTimeSeries.loader must be callable.")
        if self.attrs is None:
            self.attrs = {}
        elif not isinstance(self.attrs, dict):
            raise InvalidTimeSeries("`attrs` must be a dict.")

    def _ensure_loaded(self) -> None:
        if self._time is not None and self._values is not None:
            return

        t, v = self.loader()
        t = np.asarray(t)
        v = np.asarray(v)

        if t.ndim != 1:
            raise InvalidTimeSeries(f"`time` must be 1D, got shape {t.shape}")
        if v.ndim != 1:
            raise InvalidTimeSeries(f"`values` must be 1D, got shape {v.shape}")
        if t.size != v.size:
            raise InvalidTimeSeries(
                f"`time` and `values` must have same length, got {t.size} vs {v.size}"
            )

        if t.size > 0:
            if not np.isfinite(t).all():
                raise InvalidTimeSeries("`time` contains non-finite values (NaN/Inf).")
            dt = np.diff(t)
            if np.any(dt < 0):
                raise InvalidTimeSeries("`time` must be monotonic non-decreasing.")

        self._time = t
        self._values = v

    @property
    def time(self) -> np.ndarray:
        self._ensure_loaded()
        return self._time  # type: ignore[return-value]

    @property
    def values(self) -> np.ndarray:
        self._ensure_loaded()
        return self._values  # type: ignore[return-value]

    @property
    def n(self) -> int:
        self._ensure_loaded()
        return int(self.time.size)

    @property
    def t_start(self) -> float | None:
        if self.n == 0:
            return None
        return float(self.time[0])

    @property
    def t_end(self) -> float | None:
        if self.n == 0:
            return None
        return float(self.time[-1])

    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
    ) -> TimeSeries:
        # Keep it simple: slicing materializes.
        self._ensure_loaded()
        return TimeSeries(
            time=self.time,
            values=self.values,
            unit=self.unit,
            name=self.name,
            attrs=self.attrs.copy(),
        ).slice_time(t_min, t_max, closed=closed)

    def mean(self, *, skipna: bool = True) -> float | None:
        self._ensure_loaded()
        return TimeSeries(
            time=self.time,
            values=self.values,
            unit=self.unit,
            name=self.name,
            attrs=self.attrs.copy(),
        ).mean(skipna=skipna)

    def std(self, *, ddof: int = 0, skipna: bool = True) -> float | None:
        self._ensure_loaded()
        return TimeSeries(
            time=self.time,
            values=self.values,
            unit=self.unit,
            name=self.name,
            attrs=self.attrs.copy(),
        ).std(ddof=ddof, skipna=skipna)

    def to_numpy(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_loaded()
        if copy:
            return self.time.copy(), self.values.copy()
        return self.time, self.values