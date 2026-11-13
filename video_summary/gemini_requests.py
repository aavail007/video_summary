from __future__ import annotations

import math
import os

from google import genai
from google.genai import types


def request_timeout_seconds() -> float:
    try:
        value = float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "300"))
        if math.isfinite(value) and value > 0:
            return max(1.0, value)
    except (TypeError, ValueError):
        pass
    return 300.0


def create_gemini_client(api_key: str):
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=math.ceil(request_timeout_seconds() * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    # Standard GenAI counts total attempts, but Interactions 2.x interprets the
    # same value as EXTRA retries. The parent also normalizes 0 into 1, so a
    # zero HttpRetryOptions value alone still sends each 503 request twice.
    # Centralize this SDK bridge workaround and cover it with a real transport
    # test. Our application, not the SDK, owns the retry budget.
    config = getattr(client.interactions, "sdk_configuration", None)
    retry_config = getattr(config, "retry_config", None)
    if retry_config is not None:
        retry_config.max_retries = 0
    return client
