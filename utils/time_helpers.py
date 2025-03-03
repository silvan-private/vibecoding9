from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TimeFormatter:
    @staticmethod
    def format(seconds: float, ms_precision: bool = False) -> str:
        """Convert seconds to hh:mm:ss.ms format
        
        Args:
            seconds (float): Time in seconds
            ms_precision (bool): Whether to include milliseconds
            
        Returns:
            str: Formatted time string in HH:MM:SS[.mmm] format
            
        Raises:
            ValueError: If seconds is negative
        """
        if seconds is None:
            return "00:00:00"
            
        if seconds < 0:
            raise ValueError("Timestamp cannot be negative")
            
        try:
            # Extract hours, minutes, seconds and milliseconds
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            
            if ms_precision:
                # Format with milliseconds (3 decimal places)
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
            else:
                # Format without milliseconds
                return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"
                
        except Exception as e:
            logger.error(f"Error formatting timestamp {seconds}: {str(e)}")
            return "00:00:00"
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to mm:ss.ms format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string in mm:ss.ms format
            
        Examples:
            >>> TimeFormatter.format_timestamp(75.5)
            '01:15.500'
            >>> TimeFormatter.format_timestamp(0)
            '00:00.000'
        """
        if seconds < 0:
            raise ValueError("Timestamp cannot be negative")
            
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes:02d}:{remaining:06.3f}"
    
    @staticmethod
    def parse(time_str: str) -> float:
        """Parse a time string in HH:MM:SS format to seconds
        
        Args:
            time_str (str): Time string in HH:MM:SS format
            
        Returns:
            float: Time in seconds
            
        Raises:
            ValueError: If time string is invalid
        """
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        except ValueError as e:
            logger.error(f"Error parsing time string {time_str}: {str(e)}")
            raise

    @staticmethod
    def parse_timestamp(timestamp: str) -> float:
        """Convert mm:ss.ms format to seconds.
        
        Args:
            timestamp (str): Time string in mm:ss.ms format
            
        Returns:
            float: Time in seconds
            
        Examples:
            >>> TimeFormatter.parse_timestamp("01:15.500")
            75.5
            >>> TimeFormatter.parse_timestamp("00:00.000")
            0.0
        """
        try:
            minutes, seconds = timestamp.split(":")
            return int(minutes) * 60 + float(seconds)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}. Expected mm:ss.ms") from e

def test_time_formatter():
    """Test the TimeFormatter class with various inputs"""
    test_cases = [
        # (input_seconds, ms_precision, expected_output)
        (75.5, False, "00:01:15"),
        (75.5, True, "00:01:15.500"),
        (3661.1234, False, "01:01:01"),
        (3661.1234, True, "01:01:01.123"),
        (0, False, "00:00:00"),
        (0, True, "00:00:00.000"),
    ]
    
    for seconds, ms_precision, expected in test_cases:
        result = TimeFormatter.format(seconds, ms_precision)
        assert result == expected, f"Failed: {seconds} -> {result} != {expected}"
        
        # If we have milliseconds, test round-trip conversion
        if ms_precision:
            parsed = TimeFormatter.parse(result)
            # Allow small floating point differences
            assert abs(parsed - seconds) < 0.001, f"Round-trip failed: {seconds} -> {result} -> {parsed}"
    
    # Test error cases
    try:
        TimeFormatter.format(-1)
        assert False, "Should have raised ValueError for negative input"
    except ValueError:
        pass
        
    print("All timestamp tests passed!")

# For backwards compatibility
format_timestamp = TimeFormatter.format_timestamp

if __name__ == "__main__":
    test_time_formatter() 