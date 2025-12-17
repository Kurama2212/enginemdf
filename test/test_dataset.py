# test/test_dataset.py
import numpy as np
import pytest

from enginemdf.core import Dataset, Segment, Channel, TimeSeries, ChannelMeta, DatasetMeta
from enginemdf.core import InvalidDataset, SegmentNotFound


def _ch(name: str, t, v, unit=None):
    ts = TimeSeries(time=np.array(t, dtype=float), values=np.array(v, dtype=float), unit=unit)
    return Channel(name=name, series=ts, meta=ChannelMeta(unit=unit))


def _seg(name: str, channels: dict[str, Channel]) -> Segment:
    return Segment(name=name, channels=channels)


def test_dataset_basic_access_and_bounds():
    s1 = _seg("S1", {"a": _ch("a", [0, 1, 2], [1, 2, 3])})
    s2 = _seg("S2", {"b": _ch("b", [0.5, 1.5], [10, 20])})

    ds = Dataset(segments={"S1": s1, "S2": s2}, meta=DatasetMeta(description="x"))

    assert len(ds) == 2
    assert "S1" in ds
    assert ds["S1"].name == "S1"
    assert ds.t_start == 0.0
    assert ds.t_end == 2.0


def test_dataset_rejects_segment_key_name_mismatch():
    s1 = _seg("S1", {"a": _ch("a", [0, 1], [1, 2])})
    with pytest.raises(InvalidDataset):
        Dataset(segments={"X": s1})  # key != Segment.name


def test_dataset_getitem_missing_raises():
    ds = Dataset()
    with pytest.raises(SegmentNotFound):
        _ = ds["missing"]


def test_dataset_add_drop_select():
    ds = Dataset()
    s1 = _seg("S1", {"a": _ch("a", [0, 1], [1, 2])})

    ds2 = ds.add(s1)
    assert "S1" in ds2

    # collision without overwrite
    with pytest.raises(InvalidDataset):
        _ = ds2.add(s1, overwrite=False)

    # overwrite ok
    ds3 = ds2.add(s1, overwrite=True)
    assert len(ds3) == 1

    ds4 = ds3.drop("S1")
    assert len(ds4) == 0

    with pytest.raises(SegmentNotFound):
        _ = ds4.drop("S1")

    # select
    s2 = _seg("S2", {"b": _ch("b", [0], [1])})
    ds5 = Dataset(segments={"S1": s1, "S2": s2})
    ds6 = ds5.select(["S2"])
    assert list(ds6.keys()) == ["S2"]


def test_dataset_slice_time_and_drop_empty():
    s1 = _seg("S1", {"a": _ch("a", [0, 1, 2, 3], [10, 20, 30, 40])})
    s2 = _seg("S2", {"b": _ch("b", [0.5, 1.5], [1, 2])})
    ds = Dataset(segments={"S1": s1, "S2": s2})

    out = ds.slice_time(1.0, 2.0, closed="both")
    assert np.allclose(out["S1"]["a"].time, [1.0, 2.0])
    assert np.allclose(out["S2"]["b"].time, [1.5])

    out2 = ds.slice_time(10.0, 20.0, drop_empty_channels=True, drop_empty_segments=True)
    assert len(out2) == 0


def test_dataset_rename_segment():
    s1 = _seg("S1", {"a": _ch("a", [0, 1], [1, 2])})
    ds = Dataset(segments={"S1": s1})

    ds2 = ds.rename_segment("S1", "S1_new")
    assert "S1_new" in ds2
    assert ds2["S1_new"].name == "S1_new"

    with pytest.raises(SegmentNotFound):
        _ = ds.rename_segment("missing", "x")


def test_dataset_merge():
    s1 = _seg("S1", {"a": _ch("a", [0], [1])})
    s2 = _seg("S2", {"b": _ch("b", [0], [2])})

    d1 = Dataset(segments={"S1": s1})
    d2 = Dataset(segments={"S2": s2})

    d3 = d1.merge(d2)
    assert set(d3.keys()) == {"S1", "S2"}

    # collision
    d4 = Dataset(segments={"S1": s1})
    with pytest.raises(InvalidDataset):
        _ = d1.merge(d4, overwrite=False)

    d5 = d1.merge(d4, overwrite=True)
    assert set(d5.keys()) == {"S1"}