# core/timeseries.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class TimeSeriesError(Exception):
    """Base error for TimeSeries-related failures."""


class InvalidTimeSeries(TimeSeriesError):
    """Raised when inputs are invalid (shape, dtype, ordering, etc.)."""


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """
    Immutable time series: 1D time vector + 1D values vector.

    Design goals:
    - predictable & safe: strong validation at construction
    - fast: uses numpy arrays; slicing returns views when possible
    - minimal API: slicing, basic stats, convenience properties
    """

    time: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    unit: str | None = None
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        t = np.asarray(self.time)
        v = np.asarray(self.values)

        # Force 1D
        if t.ndim != 1:
            raise InvalidTimeSeries(f"`time` must be 1D, got shape {t.shape}")
        if v.ndim != 1:
            raise InvalidTimeSeries(f"`values` must be 1D, got shape {v.shape}")

        # Length match
        if t.size != v.size:
            raise InvalidTimeSeries(
                f"`time` and `values` must have same length, got {t.size} vs {v.size}"
            )

        # Empty is allowed? I'd say yes (helps in pipelines), but then slicing/stats behave.
        # However time ordering checks would be ambiguous; handle gracefully.
        if t.size > 0:
            # time must be finite
            if not np.isfinite(t).all():
                raise InvalidTimeSeries("`time` contains non-finite values (NaN/Inf).")

            # monotonic non-decreasing (allow duplicates)
            dt = np.diff(t)
            if np.any(dt < 0):
                raise InvalidTimeSeries("`time` must be monotonic non-decreasing.")

        # Store as numpy arrays (keep dtypes)
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "values", v)

        # Normalize attrs to dict
        if self.attrs is None:
            object.__setattr__(self, "attrs", {})
        elif not isinstance(self.attrs, dict):
            raise InvalidTimeSeries("`attrs` must be a dict.")

    @property
    def n(self) -> int:
        return int(self.time.size)

    @property
    def t_start(self) -> float | None:
        return None if self.n == 0 else float(self.time[0])

    @property
    def t_end(self) -> float | None:
        return None if self.n == 0 else float(self.time[-1])

    def is_uniform(self, rtol: float = 1e-6, atol: float = 1e-12) -> bool:
        """Return True if dt is (approximately) constant (empty/1-sample => True)."""
        if self.n <= 2:
            return True
        dt = np.diff(self.time)
        return bool(np.allclose(dt, dt[0], rtol=rtol, atol=atol))

    def slice_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        *,
        closed: str = "both",
    ) -> "TimeSeries":
        """
        Slice by time range using boolean mask.

        closed:
          - "both"  : include endpoints (>= t_min and <= t_max)
          - "left"  : include left, exclude right (>= t_min and <  t_max)
          - "right" : exclude left, include right (>  t_min and <= t_max)
          - "neither": exclude both (> t_min and < t_max)
        """
        if closed not in {"both", "left", "right", "neither"}:
            raise ValueError("closed must be one of: both, left, right, neither")

        if self.n == 0:
            return self  # nothing to slice

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

    def with_unit(self, unit: str | None) -> "TimeSeries":
        return TimeSeries(
            time=self.time,
            values=self.values,
            unit=unit,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def with_name(self, name: str | None) -> "TimeSeries":
        return TimeSeries(
            time=self.time,
            values=self.values,
            unit=self.unit,
            name=name,
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
        """
        Return (time, values). If copy=True, returns copies.
        """
        if copy:
            return self.time.copy(), self.values.copy()
        return self.time, self.values