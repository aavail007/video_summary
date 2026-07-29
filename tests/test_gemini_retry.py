import pytest

from video_summary import gemini_retry


class FakeRateLimitError(RuntimeError):
    code = 429


def test_retries_after_server_requested_delay(monkeypatch) -> None:
    attempts = 0
    waits: list[float] = []
    statuses: list[str] = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeRateLimitError("Please retry in 2.5s.")
        return "ok"

    monkeypatch.setattr(gemini_retry.time, "sleep", waits.append)

    result = gemini_retry.call_gemini_with_retry(
        operation,
        progress_callback=statuses.append,
    )

    assert result == "ok"
    assert attempts == 2
    assert waits == [12.5]
    assert "等待 13 秒" in statuses[0]


def test_does_not_retry_non_rate_limit_errors() -> None:
    with pytest.raises(ValueError, match="invalid request"):
        gemini_retry.call_gemini_with_retry(
            lambda: (_ for _ in ()).throw(ValueError("invalid request"))
        )
