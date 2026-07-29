import pytest

from video_summary import gemini_retry


@pytest.fixture(autouse=True)
def disable_gemini_throttle_during_tests(monkeypatch):
    monkeypatch.setenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(gemini_retry, "_LAST_REQUEST_STARTED", 0.0)
    gemini_retry._MODEL_BLOCKED_UNTIL.clear()
