"""
Retry with exponential backoff for calls to external services.

Only retries errors that are likely to be temporary (rate limits, server
hiccups, network blips). Errors like a bad API key fail immediately —
retrying them wastes time and can't succeed.
"""

import time

TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "timed out","rate limit", "connection", "temporarily")

def is_transient(error: Exception) -> bool:
    """Guess whether an error is temporary by looking at its message."""
    text=f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)

def with_retry(fn, *args, attempts: int = 3, base_delay: float = 1.0, **kwargs):
    """
    Call fn(*args, **kwargs). On a transient error, wait and try again,
    doubling the wait each time. Give up after `attempts` tries.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt==attempts or not is_transient(e):
                raise
            delay=base_delay*(2**(attempt-1))
            print(f"[retry] {type(e).__name__}, attempt {attempt}/{attempts}, "
                  f"waiting {delay:.0f}s")
            time.sleep(delay)