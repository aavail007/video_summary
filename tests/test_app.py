import app
from fastapi.testclient import TestClient


def _run_process(monkeypatch, source_type: str, job_id="", action=app.ACTION_NEW) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return "transcript", "summary", ["result.md"], "done", "20260812-120000-a1b2c3d4"

    monkeypatch.setattr(app, "run_pipeline", fake_run_pipeline)
    if job_id:
        monkeypatch.setattr(app, "read_job_info", lambda settings, reference: {"job_id": "20261010-205946-4b960c2c"})
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
        job_id=job_id,
        action=action,
        progress=lambda *args, **kwargs: None,
    )

    assert result == (
        "done",
        "transcript",
        "summary",
        ["result.md"],
        "20260812-120000-a1b2c3d4",
    )
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


def test_resume_from_folder_ignores_stale_upload_and_url(monkeypatch):
    captured = _run_process(monkeypatch, app.SOURCE_UPLOAD,
                            job_id=r"data\jobs\20261010-205946-4b960c2c", action=app.ACTION_RESUME)
    assert captured["resume_job_id"] == "20261010-205946-4b960c2c"
    assert captured["uploaded_path"] is None
    assert captured["youtube_url"] == ""
    assert not captured["summary_only"]


def test_loading_job_restores_settings_without_restoring_api_key(monkeypatch):
    monkeypatch.setattr(app, "read_job_info", lambda *args: {
        "job_id": "20261010-205946-4b960c2c", "source_name": "課程",
        "transcript_ready": False, "completed_chunks": 7, "total_chunks": 11,
        "source_exists": True,
        "options": {"provider": "Gemini", "gemini_model": "gemini-test", "language_label": "繁體中文"},
    })
    result = app.load_saved_job(r"data\jobs\20261010-205946-4b960c2c")
    assert result[0] == "20261010-205946-4b960c2c"
    assert "7/11" in result[1]
    assert result[2]["value"] == "Gemini"
    assert result[6]["value"] == "gemini-test"
    assert result[-1]["value"] == ""


def test_loading_missing_job_keeps_current_selection(monkeypatch):
    def missing(*args):
        raise ValueError("找不到指定工作")
    monkeypatch.setattr(app, "read_job_info", missing)
    result = app.load_saved_job("missing")
    assert len(result) == 11
    assert "value" not in result[0]
    assert "無法載入工作" in result[1]


def test_server_exposes_live_page_before_gradio_mount() -> None:
    client = TestClient(app.create_server_app())

    live_response = client.get("/live")
    config_response = client.get("/api/live/config")

    assert live_response.status_code == 200
    assert "麥克風與螢幕音訊逐字稿" in live_response.text
    assert config_response.status_code == 200
    assert "gemini_model" in config_response.json()


def test_summary_page_uses_shared_navigation_and_three_card_flow() -> None:
    demo = app.build_app()
    components = demo.config["components"]
    classes = [
        item.get("props", {}).get("elem_classes", [])
        for item in components
    ]
    html_values = [
        item.get("props", {}).get("value", "")
        for item in components
        if item.get("type") == "html"
    ]

    assert sum("app-card" in item for item in classes) == 3
    assert sum("page-shell" in item for item in classes) == 1
    assert any("app-topbar-inner" in value for value in html_values)
    assert any("影音檔案／YouTube 摘要" in value for value in html_values)
