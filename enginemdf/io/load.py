# enginemdf/io/load.py
from __future__ import annotations

from pathlib import Path

from enginemdf.io.mdf_reader import AsammdfReader
from enginemdf.core import Dataset, Segment, Channel, TimeSeries, ChannelMeta, SegmentMeta, DatasetMeta


def load_mdf(path: str) -> Dataset:
    reader = AsammdfReader(path)

    channels = {}
    for raw_ch in reader.list_channels():
        t, v = raw_ch.load()
        ts = TimeSeries(time=t, values=v, unit=raw_ch.unit, name=raw_ch.logical_name)
        ch = Channel(
            name=raw_ch.logical_name,
            series=ts,
            meta=ChannelMeta(unit=raw_ch.unit, source=f"MDF:{raw_ch.key}"),
        )
        channels[ch.name] = ch

    seg_name = Path(path).stem
    seg = Segment(
        name=seg_name,
        channels=channels,
        meta=SegmentMeta(kind="mdf_import", source=path),
    )

    ds = Dataset(
        segments={seg.name: seg},
        meta=DatasetMeta(source=path),
    )
    return ds