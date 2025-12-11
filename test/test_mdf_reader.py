import pytest
import numpy as np
from pathlib import Path

from enginemdf.io.mdf_reader import (
    AsammdfReader,
    RawChannelInfo,
    RawChannelData,
    MetaChannelInfo,
    _parse_key_and_index,
    _extract_measurement_name,
)


# Path to the test MDF file
TEST_MDF_FILE = Path(__file__).parent / "files" / "M21J_250806_M21J21_02B01_01_I5.24455.mf4"


@pytest.fixture
def mdf_reader():
    """Create an AsammdfReader instance with the test MDF file."""
    return AsammdfReader(str(TEST_MDF_FILE))


class TestParsingFunctions:
    """Test the helper parsing functions."""

    def test_parse_key_and_index_with_index(self):
        """Test parsing measurement names with indices."""
        key, idx = _parse_key_and_index("RecResult[1]")
        assert key == "RecResult"
        assert idx == 1

        key, idx = _parse_key_and_index("D[3]")
        assert key == "D"
        assert idx == 3

    def test_parse_key_and_index_without_index(self):
        """Test parsing measurement names without indices."""
        key, idx = _parse_key_and_index("Foo")
        assert key == "Foo"
        assert idx is None

    def test_parse_key_and_index_edge_cases(self):
        """Test edge cases for parsing."""
        key, idx = _parse_key_and_index("RecResult[10]")
        assert key == "RecResult"
        assert idx == 10

    def test_extract_measurement_name(self):
        """Test extracting measurement name from source path."""
        result = _extract_measurement_name("some/path/RecResult[1]")
        assert result == "RecResult[1]"

        result = _extract_measurement_name("D[2]")
        assert result == "D[2]"

        result = _extract_measurement_name("")
        assert result == ""

        # Test with backslashes (Windows-style paths)
        result = _extract_measurement_name("some\\path\\RecResult[3]")
        assert result == "RecResult[3]"


class TestAsammdfReader:
    """Test the AsammdfReader class."""

    def test_initialization(self, mdf_reader):
        """Test that the reader initializes correctly."""
        assert mdf_reader is not None
        assert mdf_reader._mdf is not None
        assert isinstance(mdf_reader._channels_by_key_name, dict)
        assert isinstance(mdf_reader._logical_index, dict)

    def test_list_channels(self, mdf_reader):
        """Test listing available channels."""
        channels = mdf_reader.list_channels()
        
        assert isinstance(channels, list)
        assert len(channels) > 0
        
        # Check that all returned items are RawChannelInfo
        for ch in channels:
            assert isinstance(ch, RawChannelInfo)
            assert ch.logical_name is not None
            assert ch.key is not None
            assert isinstance(ch.segments, list)
            assert len(ch.segments) > 0

    def test_list_channels_unique_names(self, mdf_reader):
        """Test that list_channels returns unique logical names."""
        channels = mdf_reader.list_channels()
        names = [ch.logical_name for ch in channels]
        
        # Each logical name should appear only once (default key selected)
        assert len(names) == len(set(names))

    def test_read_single_channel(self, mdf_reader):
        """Test reading a single channel."""
        channels = mdf_reader.list_channels()
        
        # Get the first available channel name
        if len(channels) > 0:
            channel_name = channels[0].logical_name
            
            result = mdf_reader.read_channels([channel_name])
            
            assert channel_name in result
            assert isinstance(result[channel_name], RawChannelData)
            assert isinstance(result[channel_name].time, np.ndarray)
            assert isinstance(result[channel_name].values, np.ndarray)
            assert len(result[channel_name].time) == len(result[channel_name].values)
            assert len(result[channel_name].time) > 0

    def test_read_multiple_channels(self, mdf_reader):
        """Test reading multiple channels at once."""
        channels = mdf_reader.list_channels()
        
        # Read up to 3 channels
        channel_names = [ch.logical_name for ch in channels[:min(3, len(channels))]]
        
        if len(channel_names) > 0:
            result = mdf_reader.read_channels(channel_names)
            
            assert len(result) == len(channel_names)
            for name in channel_names:
                assert name in result
                assert isinstance(result[name].time, np.ndarray)
                assert isinstance(result[name].values, np.ndarray)

    def test_read_nonexistent_channel(self, mdf_reader):
        """Test that reading a nonexistent channel raises KeyError."""
        with pytest.raises(KeyError, match="Channel .* not found"):
            mdf_reader.read_channels(["nonexistent_channel_xyz"])

    def test_read_channels_with_time_window(self, mdf_reader):
        """Test reading channels with time window filtering."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            channel_name = channels[0].logical_name
            
            # Read full data first
            full_data = mdf_reader.read_channels([channel_name])
            full_time = full_data[channel_name].time
            
            if len(full_time) > 0:
                # Define a time window in the middle
                t_min, t_max = full_time.min(), full_time.max()
                t_range = t_max - t_min
                
                start_time = t_min + t_range * 0.25
                end_time = t_min + t_range * 0.75
                
                # Read with time window
                windowed_data = mdf_reader.read_channels(
                    [channel_name],
                    start_time=start_time,
                    end_time=end_time
                )
                
                windowed_time = windowed_data[channel_name].time
                
                # Check that all times are within the window
                assert len(windowed_time) <= len(full_time)
                if len(windowed_time) > 0:
                    assert windowed_time.min() >= start_time
                    assert windowed_time.max() <= end_time

    def test_read_channels_with_start_time_only(self, mdf_reader):
        """Test reading channels with only start_time specified."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            channel_name = channels[0].logical_name
            
            full_data = mdf_reader.read_channels([channel_name])
            full_time = full_data[channel_name].time
            
            if len(full_time) > 0:
                start_time = full_time.min() + (full_time.max() - full_time.min()) * 0.5
                
                filtered_data = mdf_reader.read_channels(
                    [channel_name],
                    start_time=start_time
                )
                
                filtered_time = filtered_data[channel_name].time
                
                if len(filtered_time) > 0:
                    assert filtered_time.min() >= start_time

    def test_read_channels_with_end_time_only(self, mdf_reader):
        """Test reading channels with only end_time specified."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            channel_name = channels[0].logical_name
            
            full_data = mdf_reader.read_channels([channel_name])
            full_time = full_data[channel_name].time
            
            if len(full_time) > 0:
                end_time = full_time.min() + (full_time.max() - full_time.min()) * 0.5
                
                filtered_data = mdf_reader.read_channels(
                    [channel_name],
                    end_time=end_time
                )
                
                filtered_time = filtered_data[channel_name].time
                
                if len(filtered_time) > 0:
                    assert filtered_time.max() <= end_time


    def test_list_metadata_channels_basic(self, mdf_reader):
        """Test that metadata channels can be listed and have consistent structure."""
        meta_channels = mdf_reader.list_metadata_channels()

        # Should always return a list (possibly empty)
        assert isinstance(meta_channels, list)

        for ch in meta_channels:
            assert isinstance(ch, MetaChannelInfo)
            # Basic attributes must exist and have reasonable types
            assert isinstance(ch.group_index, int)
            assert isinstance(ch.channel_index, int)
            assert isinstance(ch.name, str)
            assert isinstance(ch.n_samples, int)

            # Cross-check the heuristic with the underlying MDF
            sig = mdf_reader._mdf.get(group=ch.group_index, index=ch.channel_index)
            # Either zero samples, or timestamps array is empty
            if ch.n_samples == 0:
                assert sig.timestamps.size == 0 or len(sig.timestamps) == 0
            else:
                assert sig.timestamps.size == 0

    def test_metadata_channels_not_exposed_as_logical(self, mdf_reader):
        """Ensure that metadata channels do not appear inside logical time-series channels."""
        meta_channels = mdf_reader.list_metadata_channels()
        logical_channels = mdf_reader.list_channels()

        # For each metadata channel, ensure it is not used as a segment in any logical channel.
        for meta in meta_channels:
            for ch_info in logical_channels:
                for seg in ch_info.segments:
                    assert not (
                        seg.group_index == meta.group_index
                        and seg.channel_index == meta.channel_index
                    )


class TestRawChannelInfo:
    """Test the RawChannelInfo class and its load method."""

    def test_load_without_measure_id(self, mdf_reader):
        """Test loading channel data without measurement ID."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            ch_info = channels[0]
            time, values = ch_info.load(with_measure_id=False)
            
            assert isinstance(time, np.ndarray)
            assert isinstance(values, np.ndarray)
            assert len(time) == len(values)
            assert len(time) > 0

    def test_load_with_measure_id(self, mdf_reader):
        """Test loading channel data with measurement ID."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            ch_info = channels[0]
            time, values, meas_id = ch_info.load(with_measure_id=True)
            
            assert isinstance(time, np.ndarray)
            assert isinstance(values, np.ndarray)
            assert isinstance(meas_id, np.ndarray)
            assert len(time) == len(values) == len(meas_id)
            assert meas_id.dtype == np.int32

    def test_channel_info_attributes(self, mdf_reader):
        """Test that RawChannelInfo has the expected attributes."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            ch_info = channels[0]
            
            assert hasattr(ch_info, 'logical_name')
            assert hasattr(ch_info, 'key')
            assert hasattr(ch_info, 'segments')
            assert hasattr(ch_info, 'unit')
            assert hasattr(ch_info, 'dtype')
            
            # Segments should be a non-empty list
            assert isinstance(ch_info.segments, list)
            assert len(ch_info.segments) > 0
            
            # Each segment should have the required attributes
            for seg in ch_info.segments:
                assert hasattr(seg, 'measurement_name')
                assert hasattr(seg, 'key')
                assert hasattr(seg, 'channel_name')
                assert hasattr(seg, 'loader')
                assert callable(seg.loader)


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self, mdf_reader):
        """Test the complete workflow: list channels, read data, verify consistency."""
        # List all channels
        channels = mdf_reader.list_channels()
        assert len(channels) > 0
        
        # Read all available channels
        channel_names = [ch.logical_name for ch in channels]
        data = mdf_reader.read_channels(channel_names)
        
        assert len(data) == len(channel_names)
        
        # Verify each channel's data
        for name in channel_names:
            assert name in data
            channel_data = data[name]
            assert len(channel_data.time) > 0
            assert len(channel_data.time) == len(channel_data.values)
            
            # Time should be monotonically increasing (or at least non-decreasing)
            if len(channel_data.time) > 1:
                time_diffs = np.diff(channel_data.time)
                assert np.all(time_diffs >= 0), f"Time not monotonic for channel {name}"

    def test_data_consistency_across_reads(self, mdf_reader):
        """Test that reading the same channel twice gives identical results."""
        channels = mdf_reader.list_channels()
        
        if len(channels) > 0:
            channel_name = channels[0].logical_name
            
            # Read the same channel twice
            data1 = mdf_reader.read_channels([channel_name])
            data2 = mdf_reader.read_channels([channel_name])
            
            # Results should be identical
            np.testing.assert_array_equal(
                data1[channel_name].time,
                data2[channel_name].time
            )
            np.testing.assert_array_equal(
                data1[channel_name].values,
                data2[channel_name].values
            )
