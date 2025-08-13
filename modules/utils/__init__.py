"""
Utils module initialization
"""
# Common utility functions
def format_timestamp(timestamp):
    """Format timestamp to standard format"""
def parse_timestamp(timestamp_str):
    """Parse timestamp from standard format"""
    from datetime import datetime
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
