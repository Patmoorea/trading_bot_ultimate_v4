"""
Date and Time utilities
Version: 1.0.0
"""
from datetime import datetime
import pytz
from typing import Optional
def get_utc_now() -> datetime:
    """Get current UTC datetime"""
def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO 8601 string"""
    return dt.isoformat()
def parse_timestamp(timestamp: str) -> datetime:
    """Parse ISO 8601 timestamp string to datetime"""
    try:
        # Essayer d'abord avec le timezone
        return datetime.fromisoformat(timestamp)
    except ValueError:
        # Si pas de timezone, assumer UTC
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.replace(tzinfo=pytz.UTC)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from e
