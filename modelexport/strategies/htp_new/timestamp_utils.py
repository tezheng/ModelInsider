"""Simple timestamp formatting utility."""

from datetime import UTC, datetime


def format_timestamp_iso(epoch_time: float | None) -> str | None:
    """Format Unix epoch timestamp to ISO 8601 with Z suffix.
    
    Args:
        epoch_time: Unix epoch timestamp as float, or None
        
    Returns:
        ISO 8601 string with milliseconds and Z suffix, or None if input is None
    """
    if epoch_time is None:
        return None
    dt = datetime.fromtimestamp(epoch_time, tz=UTC)
    return dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')