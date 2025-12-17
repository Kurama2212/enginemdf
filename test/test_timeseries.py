# test/test_timeseries.py
import numpy as np
import pytest

from enginemdf.core.timeseries import TimeSeries, InvalidTimeSeries


def test_init_ok_basic():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([10.0, 20.0, 30.0])
    ts = TimeSeries(time=t, values=v, unit="rpm", name="eng_spd")

    assert ts.n == 3
    assert ts.t_start == 0.0
    assert ts.t_end == 2.0
    assert ts.unit == "rpm"
    assert ts.name == "eng_spd"


def test_init_rejects_non_1d():
    t = np.array([[0.0, 1.0]])
    v = np.array([1.0, 2.0])
    with pytest.raises(InvalidTimeSeries):
        TimeSeries(time=t, values=v)


def test_init_rejects_length_mismatch():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([1.0, 2.0])
    with pytest.raises(InvalidTimeSeries):
        TimeSeries(time=t, values=v)


def test_init_rejects_non_finite_time():
    t = np.array([0.0, np.nan, 2.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(InvalidTimeSeries):
        TimeSeries(time=t, values=v)


def test_init_rejects_non_monotonic_time():
    t = np.array([0.0, 2.0, 1.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(InvalidTimeSeries):
        TimeSeries(time=t, values=v)


def test_allows_duplicate_times_non_decreasing():
    t = np.array([0.0, 1.0, 1.0, 2.0])
    v = np.array([1.0, 2.0, 3.0, 4.0])
    ts = TimeSeries(time=t, values=v)
    assert ts.n == 4


def test_slice_time_closed_both():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([10.0, 20.0, 30.0, 40.0])
    ts = TimeSeries(time=t, values=v, unit="u")

    out = ts.slice_time(1.0, 2.0, closed="both")
    assert np.allclose(out.time, [1.0, 2.0])
    assert np.allclose(out.values, [20.0, 30.0])
    assert out.unit == "u"


def test_slice_time_closed_left():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([10.0, 20.0, 30.0, 40.0])
    ts = TimeSeries(time=t, values=v)

    out = ts.slice_time(1.0, 2.0, closed="left")  # [1.0, 2.0)
    assert np.allclose(out.time, [1.0])
    assert np.allclose(out.values, [20.0])


def test_slice_time_empty_input_returns_self():
    ts = TimeSeries(time=np.array([]), values=np.array([]))
    out = ts.slice_time(0.0, 1.0)
    assert out is ts


def test_mean_std_skipna():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([1.0, np.nan, 3.0])
    ts = TimeSeries(time=t, values=v)

    assert ts.mean(skipna=True) == 2.0
    # np.nanmean([1, nan, 3]) = 2
    s = ts.std(skipna=True)
    assert isinstance(s, float)

    # Without skipna, mean should be nan
    m2 = ts.mean(skipna=False)
    assert np.isnan(m2)


def test_to_numpy_copy_flag():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([1.0, 2.0, 3.0])
    ts = TimeSeries(time=t, values=v)

    t_view, v_view = ts.to_numpy(copy=False)
    t_cp, v_cp = ts.to_numpy(copy=True)

    assert t_view is ts.time and v_view is ts.values
    assert t_cp is not ts.time and v_cp is not ts.values
    assert np.allclose(t_cp, ts.time) and np.allclose(v_cp, ts.values)


def test_lazy_timeseries_loads_on_access_and_caches():
    from enginemdf.core.timeseries import LazyTimeSeries

    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0])

    ts = LazyTimeSeries(loader=loader, unit="u", name="x")

    # Not loaded yet
    assert calls["n"] == 0

    # First access triggers load
    assert ts.n == 3
    assert calls["n"] == 1

    # Subsequent accesses use cache
    _ = ts.values
    _ = ts.time
    assert calls["n"] == 1