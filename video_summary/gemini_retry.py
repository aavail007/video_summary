from __future__ import annotations

import math
import os
import re
import threading
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")
_RETRY_SECONDS_RE = re.compile(r"retry\s+in\s+([\d.]+)s", re.IGNORECASE)
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_STARTED = 0.0


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
    return code == 429 or (
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
) -> T:
    for attempt in range(max_attempts):
        _wait_for_rate_slot()
        try:
            return operation()
        except Exception as error:
            if not _is_rate_limit_error(error):
                raise
            if attempt >= max_attempts - 1:
                raise RuntimeError(
                    "Gemini 在多次自動等待後仍回傳 429。這通常表示同一個 "
                    "Google AI Studio 專案仍有其他流量、已達每日額度，或免費層容量不足；"
                    "請稍後再試、切換 OpenAI，或在 AI Studio 啟用計費。"
                ) from error
            delay = _retry_after_seconds(error, attempt)
            message = (
                f"Gemini 暫時達到速率限制，等待 {math.ceil(delay)} 秒後自動重試"
                f"（第 {attempt + 1}/{max_attempts - 1} 次）"
            )
            print(f"[Gemini] {message}", flush=True)
            if progress_callback:
                progress_callback(message)
            time.sleep(delay)
    raise RuntimeError("Gemini 重試流程未能完成。")
