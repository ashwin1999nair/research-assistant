from agents.retry import with_retry, is_transient
import pytest

def make_flaky(fail_times, error):
    """A fake function that fails `fail_times` times, then succeeds."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise error
        return "ok"

    return fn, calls

def test_recovers_after_transient_failures():
    fn, calls = make_flaky(2, Exception("HTTP 503 Service Unavailable"))
    assert with_retry(fn, base_delay=0) == "ok"
    assert calls["n"] == 3

def test_gives_up_after_max_attempts():
    fn, calls = make_flaky(5, Exception("HTTP 503 Service Unavailable"))
    with pytest.raises(Exception):
        with_retry(fn, attempts=3, base_delay=0)
    assert calls["n"] == 3

def test_does_not_retry_permanent_errors():
    fn, calls = make_flaky(5, Exception("HTTP 401 Unauthorized"))
    with pytest.raises(Exception):
        with_retry(fn, base_delay=0)
    assert calls["n"] == 1

def test_is_transient_uses_class_name():
    # Empty message — the only clue is the class name "TimeoutError"
    assert is_transient(TimeoutError())


def test_is_transient_uses_message():
    # Generic class name — the only clue is "429" in the message
    assert is_transient(Exception("429"))

def test_is_transient():
    assert is_transient(Exception("429 Too Many Requests"))
    assert is_transient(TimeoutError("read timed out"))
    assert not is_transient(Exception("401 Unauthorized"))