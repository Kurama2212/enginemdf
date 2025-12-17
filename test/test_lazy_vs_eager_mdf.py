from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from enginemdf.io.mdf_reader import AsammdfReader
from enginemdf.core.timeseries import TimeSeries, LazyTimeSeries


@pytest.mark.integration
def test_lazy_vs_eager_timeseries_real_mdf_giri_mot():
    """Compare LazyTimeSeries vs TimeSeries on a real MDF file and channel GIRI_MOT."""
    mdf_path = Path(__file__).parent / "files" / "mdf_test.mf4"
    if not mdf_path.exists():
        pytest.skip(f"Missing test MDF: {mdf_path}")

    reader = AsammdfReader(str(mdf_path))

    raw_channels = list(reader.list_channels())
    raw_ch = next((ch for ch in raw_channels if ch.logical_name == "GIRI_MOT"), None)

    if raw_ch is None:
        available = sorted({ch.logical_name for ch in raw_channels})
        pytest.skip(f"Channel 'GIRI_MOT' not found. Available: {available[:20]} ...")

    # EAGER baseline
    t_eager, v_eager = raw_ch.load()
    ts_eager = TimeSeries(time=t_eager, values=v_eager, unit=raw_ch.unit, name=raw_ch.logical_name)

    # LAZY implementation (wrapped loader with call counter)
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return raw_ch.load()

    ts_lazy = LazyTimeSeries(loader=loader, unit=raw_ch.unit, name=raw_ch.logical_name)

    # Not loaded yet
    assert calls["n"] == 0

    # First access triggers load
    assert ts_lazy.n == ts_eager.n
    assert calls["n"] == 1

    # Compare arrays
    t_lazy, v_lazy = ts_lazy.to_numpy(copy=False)
    assert t_lazy.shape == ts_eager.time.shape
    assert v_lazy.shape == ts_eager.values.shape

    assert np.array_equal(t_lazy, ts_eager.time)
    assert np.array_equal(v_lazy, ts_eager.values)

    # Cache check: further accesses should NOT call loader again
    _ = ts_lazy.time
    _ = ts_lazy.values
    _ = ts_lazy.to_numpy(copy=True)
    assert calls["n"] == 1


@pytest.mark.integration
def test_lazy_slice_matches_eager_slice_real_mdf_giri_mot():
    """Ensure LazyTimeSeries.slice_time() materializes and matches eager slicing results."""
    mdf_path = Path(__file__).parent / "files" / "mdf_test.mf4"
    if not mdf_path.exists():
        pytest.skip(f"Missing test MDF: {mdf_path}")

    reader = AsammdfReader(str(mdf_path))
    raw_channels = list(reader.list_channels())
    raw_ch = next((ch for ch in raw_channels if ch.logical_name == "GIRI_MOT"), None)

    if raw_ch is None:
        pytest.skip("Channel 'GIRI_MOT' not found in test MDF.")

    t, v = raw_ch.load()
    ts_eager = TimeSeries(time=t, values=v, unit=raw_ch.unit, name=raw_ch.logical_name)

    if ts_eager.n < 2:
        pytest.skip("Not enough samples for slicing test.")

    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return raw_ch.load()

    ts_lazy = LazyTimeSeries(loader=loader, unit=raw_ch.unit, name=raw_ch.logical_name)

    # pick a window based on actual data
    t_min = float(ts_eager.time[0])
    t_max = float(ts_eager.time[-1])
    mid = (t_min + t_max) / 2.0

    eager_half = ts_eager.slice_time(t_min, mid, closed="both")
    lazy_half = ts_lazy.slice_time(t_min, mid, closed="both")

    assert calls["n"] == 1
    assert np.array_equal(lazy_half.time, eager_half.time)
    assert np.array_equal(lazy_half.values, eager_half.values)