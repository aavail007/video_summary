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
_MODEL_BLOCKED_UNTIL: dict[str, float] = {}


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


def call_gemini_with_model_fallback(
    operation: Callable[[str], T],
    *,
    primary_model: str,
    progress_callback=None,
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
        )

    try:
        _wait_for_rate_slot()
        return operation(primary_model)
    except Exception as error:
        if not _is_rate_limit_error(error):
            raise

    if not fallback_model or fallback_model == primary_model:
        return call_gemini_with_retry(
            lambda: operation(primary_model),
            progress_callback=progress_callback,
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
    print(f"[Gemini] {message}", flush=True)
    if progress_callback:
        progress_callback(message)
    return call_gemini_with_retry(
        lambda: operation(fallback_model),
        progress_callback=progress_callback,
    )
