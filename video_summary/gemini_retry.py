from __future__ import annotations

import math
import os
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import copy_context
from typing import TypeVar

import httpx

from .gemini_requests import request_timeout_seconds


T = TypeVar("T")
_RETRY_SECONDS_RE = re.compile(r"retry\s+in\s+([\d.]+)s", re.IGNORECASE)
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_STARTED = 0.0
_MODEL_BLOCKED_UNTIL: dict[str, float] = {}
_STATUS_INTERVAL_SECONDS = 5.0


class GeminiRequestError(RuntimeError):
    """A bounded retry cycle has ended; only an explicit user retry should restart it."""

    def __init__(self, message: str, *, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _transient_retries() -> int:
    try:
        return min(5, max(0, int(os.getenv("GEMINI_MAX_TRANSIENT_RETRIES", "2"))))
    except ValueError:
        return 2


def _report(message: str, progress_callback=None) -> None:
    print(f"[Gemini] {message}", flush=True)
    if progress_callback:
        try:
            progress_callback(message)
        except Exception:
            # Closing/refreshing a browser must not fail the API request.
            pass


@contextmanager
def _waiting_status(label: str, progress_callback=None):
    started = time.monotonic()
    stopped = threading.Event()
    timeout = math.ceil(request_timeout_seconds())
    _report(f"{label}：等待回應（已等待 0 秒；單次逾時 {timeout} 秒）", progress_callback)

    def heartbeat():
        while not stopped.wait(_STATUS_INTERVAL_SECONDS):
            elapsed = math.floor(time.monotonic() - started)
            _report(f"{label}：等待回應（已等待 {elapsed} 秒；單次逾時 {timeout} 秒）", progress_callback)

    # Gradio's progress routing uses context variables, including the event ID.
    context = copy_context()
    worker = threading.Thread(target=context.run, args=(heartbeat,), daemon=True, name="gemini-wait-status")
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def _wait_before_retry(seconds: float, label: str, progress_callback=None) -> None:
    # Decrement explicitly so tests need not use a real clock or sleep.
    remaining = min(120.0, max(0.0, seconds))
    while remaining > 0:
        _report(f"{label}；等待 {math.ceil(remaining)} 秒後自動重試", progress_callback)
        step = min(5.0, remaining)
        time.sleep(step)
        remaining -= step


def _is_transient_error(error: Exception) -> bool:
    code = getattr(error, "code", None)
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return (
        code in (500, 502, 503, 504) or status in (500, 502, 503, 504)
        or getattr(error, "status_code", None) in (500, 502, 503, 504)
        or _is_timeout_error(error) or _has_transport_error(error)
        or bool(re.search(r"(?:error code[: ]+|status code[: ]+)(?:500|502|503|504)\b", str(error), re.IGNORECASE))
    )


def _causes(error: Exception):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = getattr(error, "__cause__", None)


def _is_timeout_error(error: Exception) -> bool:
    return any(isinstance(cause, (httpx.TimeoutException, TimeoutError)) for cause in _causes(error))


def _has_transport_error(error: Exception) -> bool:
    return any(isinstance(cause, httpx.NetworkError) for cause in _causes(error))


def _minimum_interval() -> float:
    raw = os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "5.0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def _wait_for_rate_slot() -> None:
    global _LAST_REQUEST_STARTED
    interval = _minimum_interval()
    if interval <= 0:
        return
    with _RATE_LOCK:
        remaining = interval - (time.monotonic() - _LAST_REQUEST_STARTED)
        if remaining > 0:
            time.sleep(remaining)
        _LAST_REQUEST_STARTED = time.monotonic()


def _is_rate_limit_error(error: Exception) -> bool:
    code = getattr(error, "code", None)
    message = str(error).lower()
    return code == 429 or getattr(error, "status_code", None) == 429 or (
        "429" in message
        and ("quota" in message or "rate limit" in message or "too_many_requests" in message)
    )


def _retry_after_seconds(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    retry_after = headers.get("retry-after") if headers else None
    if retry_after:
        try:
            return max(10.0, float(retry_after) + 10.0)
        except (TypeError, ValueError):
            pass

    match = _RETRY_SECONDS_RE.search(str(error))
    if match:
        return max(10.0, float(match.group(1)) + 10.0)
    return [15.0, 30.0, 60.0, 90.0, 120.0][min(attempt, 4)]


def call_gemini_with_retry(
    operation: Callable[[], T],
    *,
    progress_callback=None,
    max_attempts: int = 6,
    request_label: str = "Gemini 請求",
    retry_rate_limits: bool = True,
) -> T:
    transient_failures = 0
    transient_retries = _transient_retries()
    if max_attempts < 1:
        raise ValueError("max_attempts 必須至少為 1")
    for attempt in range(max_attempts):
        _wait_for_rate_slot()
        try:
            with _waiting_status(f"{request_label}（第 {attempt + 1} 次請求）", progress_callback):
                return operation()
        except Exception as error:
            if _is_transient_error(error):
                transient_failures += 1
                is_timeout = _is_timeout_error(error)
                cause = f"請求逾時（單次上限 {math.ceil(request_timeout_seconds())} 秒）" if is_timeout else "服務暫時無法回應（503／網路或伺服器錯誤）"
                if transient_failures > transient_retries or attempt >= max_attempts - 1:
                    message = (
                        f"{request_label}：{cause}，已停止自動重試"
                        f"（共嘗試 {attempt + 1} 次）。已完成的音訊片段仍會保留；"
                        "請稍後或切換可用模型，再續跑同一個工作。"
                    )
                    _report(message, progress_callback)
                    raise GeminiRequestError(message, status_code=504 if is_timeout else 503) from error
                delay = min(60.0, 10.0 * 2 ** (transient_failures - 1))
                _wait_before_retry(delay, f"{request_label}：{cause}（重試 {transient_failures}/{transient_retries}）", progress_callback)
                continue
            if not _is_rate_limit_error(error) or not retry_rate_limits:
                raise
            if attempt >= max_attempts - 1:
                raise GeminiRequestError(
                    "Gemini 在多次自動等待後仍回傳 429。這通常表示同一個 "
                    "Google AI Studio 專案仍有其他流量、已達每日額度，或免費層容量不足；"
                    "請稍後再試、切換 OpenAI，或在 AI Studio 啟用計費。",
                    status_code=429,
                ) from error
            delay = _retry_after_seconds(error, attempt)
            message = (
                f"{request_label}：Gemini 暫時達到速率限制"
                f"（第 {attempt + 1}/{max_attempts - 1} 次）"
            )
            _wait_before_retry(delay, message, progress_callback)
    raise RuntimeError("Gemini 重試流程未能完成。")


def call_gemini_with_model_fallback(
    operation: Callable[[str], T],
    *,
    primary_model: str,
    progress_callback=None,
    request_label: str = "Gemini 請求",
) -> T:
    """Try the selected model once, then use a separate model quota on 429."""
    fallback_model = os.getenv(
        "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite"
    ).strip()
    if (
        fallback_model
        and fallback_model != primary_model
        and _MODEL_BLOCKED_UNTIL.get(primary_model, 0.0) > time.monotonic()
    ):
        return call_gemini_with_retry(
            lambda: operation(fallback_model),
            progress_callback=progress_callback,
            request_label=f"{request_label}／{fallback_model}",
        )

    try:
        return call_gemini_with_retry(
            lambda: operation(primary_model),
            progress_callback=progress_callback,
            request_label=f"{request_label}／{primary_model}",
            retry_rate_limits=False,
        )
    except Exception as error:
        if not _is_rate_limit_error(error):
            raise

    if not fallback_model or fallback_model == primary_model:
        return call_gemini_with_retry(
            lambda: operation(primary_model),
            progress_callback=progress_callback,
            request_label=f"{request_label}／{primary_model}",
        )

    cooldown_raw = os.getenv("GEMINI_FALLBACK_COOLDOWN_SECONDS", "600").strip()
    try:
        cooldown = max(60.0, float(cooldown_raw))
    except ValueError:
        cooldown = 600.0
    _MODEL_BLOCKED_UNTIL[primary_model] = time.monotonic() + cooldown
    message = (
        f"{primary_model} 已達配額限制，自動改用 {fallback_model} 繼續處理"
    )
    _report(message, progress_callback)
    return call_gemini_with_retry(
        lambda: operation(fallback_model),
        progress_callback=progress_callback,
        request_label=f"{request_label}／{fallback_model}",
    )
