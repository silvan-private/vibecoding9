"""Tests for time formatting utilities."""
import pytest
from .time_helpers import TimeFormatter, format_timestamp

def test_format_timestamp():
    """Test basic timestamp formatting."""
    assert format_timestamp(75.5) == "01:15.500"
    assert format_timestamp(0) == "00:00.000"
    assert format_timestamp(3661.001) == "61:01.001"  # Over 60 minutes
    assert format_timestamp(0.001) == "00:00.001"     # Small value
    assert format_timestamp(59.999) == "00:59.999"    # Just under a minute

def test_format_timestamp_errors():
    """Test error handling in timestamp formatting."""
    with pytest.raises(ValueError):
        format_timestamp(-1.0)  # Negative time
        
def test_parse_timestamp():
    """Test timestamp parsing."""
    assert TimeFormatter.parse_timestamp("01:15.500") == 75.5
    assert TimeFormatter.parse_timestamp("00:00.000") == 0.0
    assert TimeFormatter.parse_timestamp("61:01.001") == 3661.001
    
def test_parse_timestamp_errors():
    """Test error handling in timestamp parsing."""
    with pytest.raises(ValueError):
        TimeFormatter.parse_timestamp("invalid")
    with pytest.raises(ValueError):
        TimeFormatter.parse_timestamp("1:2:3")  # Wrong format
    with pytest.raises(ValueError):
        TimeFormatter.parse_timestamp("")  # Empty string

def test_roundtrip():
    """Test converting back and forth between formats."""
    original = 75.5
    formatted = format_timestamp(original)
    parsed = TimeFormatter.parse_timestamp(formatted)
    assert abs(original - parsed) < 0.001  # Account for floating point precision 