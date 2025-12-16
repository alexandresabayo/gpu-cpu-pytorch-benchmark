"""Helper functions"""

from typing import Union


def format_time(seconds: Union[int, float]) -> str:
    """Convert seconds to 'Xm Ys' format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {int(secs)}s"