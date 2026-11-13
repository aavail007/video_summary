import pytest
import httpx
import threading
from contextvars import ContextVar

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
    assert waits == [5, 5, 2.5]
    assert any("等待 13 秒" in message for message in statuses)


def test_does_not_retry_non_rate_limit_errors() -> None:
    with pytest.raises(ValueError, match="invalid request"):
        gemini_retry.call_gemini_with_retry(
            lambda: (_ for _ in ()).throw(ValueError("invalid request"))
        )


def test_switches_to_fallback_model_on_first_rate_limit(monkeypatch) -> None:
    models: list[str] = []
    statuses: list[str] = []

    def operation(model: str):
        models.append(model)
        if model == "gemini-primary":
            raise FakeRateLimitError("Error code: 429 - quota exceeded")
        return "fallback result"

    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-fallback")

    result = gemini_retry.call_gemini_with_model_fallback(
        operation,
        primary_model="gemini-primary",
        progress_callback=statuses.append,
    )

    assert result == "fallback result"
    assert models == ["gemini-primary", "gemini-fallback"]
    assert any("自動改用 gemini-fallback" in message for message in statuses)

    second_result = gemini_retry.call_gemini_with_model_fallback(
        operation,
        primary_model="gemini-primary",
    )
    assert second_result == "fallback result"
    assert models == ["gemini-primary", "gemini-fallback", "gemini-fallback"]


class FakeUnavailableError(RuntimeError):
    status_code = 503


def test_503_has_bounded_retries_without_model_fallback(monkeypatch):
    models = []
    waits = []
    statuses = []

    def operation(model):
        models.append(model)
        raise FakeUnavailableError("high demand")

    monkeypatch.setenv("GEMINI_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(gemini_retry.time, "sleep", waits.append)
    with pytest.raises(gemini_retry.GeminiRequestError, match="共嘗試 3 次") as error:
        gemini_retry.call_gemini_with_model_fallback(
            operation, primary_model="primary", progress_callback=statuses.append,
            request_label="轉錄片段 8/11",
        )
    assert models == ["primary"] * 3
    assert sum(waits) == 30
    assert error.value.status_code == 503
    assert any("重試 1/2" in message for message in statuses)
    assert any("重試 2/2" in message for message in statuses)
    assert any("轉錄片段 8/11" in message and "等待回應" in message for message in statuses)


@pytest.mark.parametrize("retries,calls", [("0", 1), ("2", 3), ("invalid", 3)])
def test_wrapped_timeout_retries_are_bounded(monkeypatch, retries, calls):
    attempts = []

    def operation():
        attempts.append(1)
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout as cause:
            raise RuntimeError("SDK wrapper") from cause

    monkeypatch.setenv("GEMINI_MAX_TRANSIENT_RETRIES", retries)
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda _: None)
    with pytest.raises(gemini_retry.GeminiRequestError, match="請求逾時") as error:
        gemini_retry.call_gemini_with_retry(operation)
    assert len(attempts) == calls
    assert error.value.status_code == 504


@pytest.mark.parametrize("fails", [False, True])
def test_wait_heartbeat_inherits_context_and_stops(monkeypatch, fails):
    event_id = ContextVar("test_event_id", default=None)
    event_id.set("gradio-event")
    heartbeat_seen = threading.Event()
    callbacks = []

    def callback(message):
        callbacks.append((threading.current_thread(), event_id.get(), message))
        if threading.current_thread().name == "gemini-wait-status":
            heartbeat_seen.set()

    def operation():
        assert heartbeat_seen.wait(1)
        if fails:
            raise ValueError("invalid request")
        return "ok"

    monkeypatch.setattr(gemini_retry, "_STATUS_INTERVAL_SECONDS", 0.01)
    if fails:
        with pytest.raises(ValueError):
            gemini_retry.call_gemini_with_retry(operation, progress_callback=callback)
    else:
        assert gemini_retry.call_gemini_with_retry(operation, progress_callback=callback) == "ok"
    assert all(context == "gradio-event" for _, context, _ in callbacks)
    assert all(not worker.is_alive() for worker, _, _ in callbacks if worker.name == "gemini-wait-status")
    assert any("已等待" in message and "單次逾時" in message for _, _, message in callbacks)
