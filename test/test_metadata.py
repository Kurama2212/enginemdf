# test/test_metadata.py
import pytest

from enginemdf.core import ChannelMeta, SegmentMeta, DatasetMeta
from enginemdf.core import InvalidChannel, InvalidSegment, InvalidDataset


def test_channelmeta_accepts_dict_and_normalizes_none():
    m = ChannelMeta(unit="rpm", attrs={"k": 1})
    assert m.attrs == {"k": 1}

    m2 = ChannelMeta(attrs=None)
    assert m2.attrs == {}


def test_channelmeta_rejects_non_dict_attrs():
    with pytest.raises(InvalidChannel):
        ChannelMeta(attrs=["not", "a", "dict"])  # type: ignore[arg-type]


def test_segmentmeta_accepts_dict_and_normalizes_none():
    m = SegmentMeta(kind="power_curve", attrs={"a": 1})
    assert m.attrs == {"a": 1}

    m2 = SegmentMeta(attrs=None)
    assert m2.attrs == {}


def test_segmentmeta_rejects_non_dict_attrs():
    with pytest.raises(InvalidSegment):
        SegmentMeta(attrs="nope")  # type: ignore[arg-type]


def test_datasetmeta_accepts_dict_and_normalizes_none():
    m = DatasetMeta(description="x", attrs={"hello": "world"})
    assert m.attrs == {"hello": "world"}

    m2 = DatasetMeta(attrs=None)
    assert m2.attrs == {}


def test_datasetmeta_rejects_non_dict_attrs():
    with pytest.raises(InvalidDataset):
        DatasetMeta(attrs=123)  # type: ignore[arg-type]