# test/test_segment.py
import numpy as np
import pytest

from enginemdf.core import Segment, Channel, TimeSeries, SegmentMeta, ChannelMeta
from enginemdf.core import InvalidSegment, ChannelNotFound


def _ch(name: str, t, v, unit=None):
    ts = TimeSeries(time=np.array(t, dtype=float), values=np.array(v, dtype=float), unit=unit)
    return Channel(name=name, series=ts, meta=ChannelMeta(unit=unit))


def test_segment_basic_dict_api_and_bounds():
    ch1 = _ch("eng_spd", [0, 1, 2], [10, 20, 30], unit="rpm")
    ch2 = _ch("torque",  [0.5, 1.5], [100, 110], unit="Nm")

    seg = Segment(
        name="PowerCurve_01",
        channels={"eng_spd": ch1, "torque": ch2},
        meta=SegmentMeta(kind="power_curve"),
    )

    assert len(seg) == 2
    assert "eng_spd" in seg
    assert seg["eng_spd"].unit == "rpm"
    assert seg.t_start == 0.0
    assert seg.t_end == 2.0


def test_segment_rejects_channel_key_name_mismatch():
    ch1 = _ch("eng_spd", [0, 1], [1, 2])
    with pytest.raises(InvalidSegment):
        Segment(name="X", channels={"spd": ch1})  # key != Channel.name


def test_segment_getitem_missing_raises_channelnotfound():
    seg = Segment(name="X", channels={})
    with pytest.raises(ChannelNotFound):
        _ = seg["missing"]


def test_segment_add_and_drop():
    seg = Segment(name="X", channels={})
    ch1 = _ch("eng_spd", [0, 1], [10, 20])

    seg2 = seg.add(ch1)
    assert "eng_spd" in seg2
    assert len(seg2) == 1

    # add without overwrite should fail
    with pytest.raises(InvalidSegment):
        _ = seg2.add(ch1, overwrite=False)

    # overwrite ok
    seg3 = seg2.add(ch1, overwrite=True)
    assert len(seg3) == 1

    # drop ok
    seg4 = seg3.drop("eng_spd")
    assert len(seg4) == 0

    # drop missing raises by default
    with pytest.raises(ChannelNotFound):
        _ = seg4.drop("eng_spd")


def test_segment_select_missing_modes():
    ch1 = _ch("a", [0, 1], [1, 2])
    ch2 = _ch("b", [0, 1], [3, 4])
    seg = Segment(name="X", channels={"a": ch1, "b": ch2})

    sel = seg.select(["b"])
    assert list(sel.keys()) == ["b"]

    with pytest.raises(ChannelNotFound):
        _ = seg.select(["c"])

    sel2 = seg.select(["c", "a"], missing="ignore")
    assert list(sel2.keys()) == ["a"]


def test_segment_slice_time_and_drop_empty():
    ch1 = _ch("a", [0, 1, 2, 3], [10, 20, 30, 40])
    ch2 = _ch("b", [0.5, 1.5], [1, 2])
    seg = Segment(name="X", channels={"a": ch1, "b": ch2})

    out = seg.slice_time(1.0, 2.0, closed="both", drop_empty=False)
    assert np.allclose(out["a"].time, [1.0, 2.0])
    assert np.allclose(out["b"].time, [1.5])  # 0.5 excluded

    out2 = seg.slice_time(10.0, 20.0, drop_empty=True)
    assert len(out2) == 0


def test_segment_rename_preserves_channels():
    ch1 = _ch("a", [0, 1], [1, 2])
    seg = Segment(name="Old", channels={"a": ch1}, meta=SegmentMeta(description="d", attrs={"k": 1}))

    seg2 = seg.rename("New")
    assert seg2.name == "New"
    assert "a" in seg2
    assert seg2.meta.description == "d"
    assert seg2.meta.attrs == {"k": 1}
    assert seg2.meta.attrs is not seg.meta.attrs