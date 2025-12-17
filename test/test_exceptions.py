# test/test_exceptions.py
import pytest

from enginemdf.core import (
    CoreError,
    InvalidTimeSeries,
    InvalidChannel,
    InvalidSegment,
    InvalidDataset,
    ChannelNotFound,
    SegmentNotFound,
)


def test_exception_inheritance_validation():
    assert issubclass(InvalidTimeSeries, CoreError)
    assert issubclass(InvalidChannel, CoreError)
    assert issubclass(InvalidSegment, CoreError)
    assert issubclass(InvalidDataset, CoreError)


def test_exception_inheritance_lookup_keyerror():
    assert issubclass(ChannelNotFound, KeyError)
    assert issubclass(ChannelNotFound, CoreError)
    assert issubclass(SegmentNotFound, KeyError)
    assert issubclass(SegmentNotFound, CoreError)


def test_lookup_errors_can_be_raised_and_caught_as_keyerror():
    with pytest.raises(KeyError):
        raise ChannelNotFound("eng_spd")

    with pytest.raises(KeyError):
        raise SegmentNotFound("PowerCurve_01")