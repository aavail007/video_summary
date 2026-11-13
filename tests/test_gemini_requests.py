import httpx
import pytest

from video_summary import gemini_requests, gemini_retry


@pytest.mark.parametrize("value,expected", [
    ("45", 45), ("0", 300), ("-1", 300), ("nan", 300),
    ("inf", 300), ("invalid", 300), ("0.5", 1),
])
def test_timeout_configuration(monkeypatch, value, expected):
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_SECONDS", value)
    assert gemini_requests.request_timeout_seconds() == expected


@pytest.mark.parametrize("failure", ["503", "timeout"])
def test_real_sdk_disables_hidden_retries_and_has_finite_timeout(monkeypatch, failure):
    requests = []
    original_client = gemini_requests.genai.Client

    def handle(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("mock timeout", request=request)
        return httpx.Response(503, json={"error": {
            "code": "unavailable", "message": "high demand",
        }})

    def client_with_mock_transport(**kwargs):
        options = kwargs["http_options"]
        assert options.timeout == 45000
        assert options.retry_options.attempts == 1
        kwargs["http_options"] = options.model_copy(update={
            "client_args": {"transport": httpx.MockTransport(handle)},
        })
        return original_client(**kwargs)

    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("GEMINI_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setattr(gemini_retry.time, "sleep", lambda _: None)
    monkeypatch.setattr(gemini_requests.genai, "Client", client_with_mock_transport)
    client = gemini_requests.create_gemini_client("test-key-not-real")
    try:
        with pytest.raises(gemini_retry.GeminiRequestError) as error:
            gemini_retry.call_gemini_with_retry(lambda: client.interactions.create(
                model="test-model", input="test", timeout=gemini_requests.request_timeout_seconds(),
            ))
        assert error.value.status_code == (504 if failure == "timeout" else 503)
        assert len(requests) == 3
        assert all(all(value == 45 for value in request.extensions["timeout"].values()) for request in requests)
    finally:
        client.close()
