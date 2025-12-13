# test/test_channel.py
import numpy as np
import pytest

from enginemdf.core.timeseries import TimeSeries
from enginemdf.core.channel import Channel, ChannelMeta, InvalidChannel


def test_channel_basic_accessors():
    ts = TimeSeries(time=np.array([0.0, 1.0]), values=np.array([10.0, 20.0]), unit="rpm")
    ch = Channel(name="eng_spd", series=ts)

    assert ch.name == "eng_spd"
    assert ch.n == 2
    assert ch.t_start == 0.0
    assert ch.t_end == 1.0
    assert ch.unit == "rpm"
    assert np.allclose(ch.values, [10.0, 20.0])


def test_channel_rejects_empty_name():
    ts = TimeSeries(time=np.array([0.0]), values=np.array([1.0]))
    with pytest.raises(InvalidChannel):
        Channel(name="   ", series=ts)


def test_channel_unit_precedence_meta_over_series():
    ts = TimeSeries(time=np.array([0.0, 1.0]), values=np.array([1.0, 2.0]), unit="A")
    ch = Channel(
        name="x",
        series=ts,
        meta=ChannelMeta(unit="B", description="desc"),
    )
    assert ch.unit == "B"


def test_channel_inherits_unit_from_series_when_meta_unit_none():
    ts = TimeSeries(time=np.array([0.0, 1.0]), values=np.array([1.0, 2.0]), unit="Nm")
    meta = ChannelMeta(unit=None, description="torque", attrs={"k": 1})
    ch = Channel(name="torque", series=ts, meta=meta)

    assert ch.unit == "Nm"
    assert ch.meta.attrs == {"k": 1}
    assert ch.meta.attrs is not meta.attrs  # copied in __post_init__


def test_slice_time_returns_channel_and_keeps_meta_copy():
    ts = TimeSeries(
        time=np.array([0.0, 1.0, 2.0, 3.0]),
        values=np.array([10.0, 20.0, 30.0, 40.0]),
        unit="u",
    )
    ch = Channel(name="x", series=ts, meta=ChannelMeta(description="hello", attrs={"a": 1}))

    out = ch.slice_time(1.0, 2.0)
    assert out.name == "x"
    assert np.allclose(out.time, [1.0, 2.0])
    assert np.allclose(out.values, [20.0, 30.0])
    assert out.meta.description == "hello"
    assert out.meta.attrs == {"a": 1}
    assert out.meta.attrs is not ch.meta.attrs


def test_rename_and_with_unit():
    ts = TimeSeries(time=np.array([0.0, 1.0]), values=np.array([1.0, 2.0]))
    ch = Channel(name="a", series=ts, meta=ChannelMeta(unit="X", attrs={"k": 1}))

    ch2 = ch.rename("b")
    assert ch2.name == "b"
    assert ch2.unit == "X"
    assert ch2.meta.attrs == {"k": 1}
    assert ch2.meta.attrs is not ch.meta.attrs

    ch3 = ch.with_unit("Y")
    assert ch3.name == "a"
    assert ch3.unit == "Y"
    assert ch3.meta.attrs == {"k": 1}