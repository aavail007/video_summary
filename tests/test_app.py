import app


def _run_process(monkeypatch, source_type: str) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return "transcript", "summary", ["result.md"], "done"

    monkeypatch.setattr(app, "run_pipeline", fake_run_pipeline)
    result = app.process(
        source_type,
        "Gemini",
        "lecture.mp4",
        "https://youtu.be/stale-url",
        "test-key",
        "自動偵測",
        "gpt-4o-transcribe",
        None,
        None,
        "low",
        "一般重點摘要",
        "",
        "不分析投影片，只分析音訊",
        None,
        None,
        True,
        progress=lambda *args, **kwargs: None,
    )

    assert result == ("done", "transcript", "summary", ["result.md"])
    return captured


def test_upload_source_ignores_hidden_youtube_url(monkeypatch) -> None:
    captured = _run_process(monkeypatch, app.SOURCE_UPLOAD)

    assert captured["uploaded_path"] == "lecture.mp4"
    assert captured["youtube_url"] == ""
    assert captured["summary_model"] == app.settings.summary_model
    assert captured["gemini_model"] == app.settings.gemini_model


def test_youtube_source_ignores_hidden_uploaded_file(monkeypatch) -> None:
    captured = _run_process(monkeypatch, app.SOURCE_YOUTUBE)

    assert captured["uploaded_path"] is None
    assert captured["youtube_url"] == "https://youtu.be/stale-url"
