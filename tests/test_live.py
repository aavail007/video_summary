from pathlib import Path

import pytest

from video_summary import live
from video_summary.config import Settings
from video_summary.models import TranscriptResult, TranscriptSegment


def test_transcribe_live_chunk_uses_gemini_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        jobs_dir=tmp_path / "data" / "jobs",
        gemini_api_key="configured-key",
        gemini_model="gemini-test",
    )
    captured: dict[str, object] = {}

    def fake_split_audio(source: Path, chunks_dir: Path, chunk_seconds: int):
        assert source.suffix == ".webm"
        assert source.read_bytes() == b"recorded-audio"
        chunks_dir.mkdir(parents=True)
        chunk = chunks_dir / "chunk_000.mp3"
        chunk.write_bytes(b"mp3")
        return [chunk]

    def fake_transcribe(chunks, **kwargs):
        captured.update(kwargs)
        return TranscriptResult(
            source_name="即時收音",
            language="zh",
            model=kwargs["model"],
            segments=[TranscriptSegment(start=0, end=2, text="測試逐字稿")],
        )

    monkeypatch.setattr(live, "split_audio", fake_split_audio)
    monkeypatch.setattr(live, "transcribe_chunks_gemini", fake_transcribe)

    result = live.transcribe_live_chunk(
        b"recorded-audio",
        content_type="audio/webm;codecs=opus",
        provider="gemini",
        api_key="",
        language_label="繁體中文",
        gemini_model="",
        openai_model="",
        settings=settings,
    )

    assert result["text"] == "測試逐字稿"
    assert result["model"] == "gemini-test"
    assert captured["api_key"] == "configured-key"
    assert not list(settings.data_dir.glob("live-*"))


def test_transcribe_live_chunk_requires_api_key(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        jobs_dir=tmp_path / "data" / "jobs",
        gemini_api_key="",
    )

    with pytest.raises(live.LiveTranscriptionError, match="API Key"):
        live.transcribe_live_chunk(
            b"audio",
            content_type="audio/webm",
            provider="gemini",
            api_key="",
            language_label="繁體中文",
            gemini_model="gemini-test",
            openai_model="",
            settings=settings,
        )


def test_live_assets_include_microphone_screen_and_markdown_controls() -> None:
    html = (live.LIVE_ASSETS_DIR / "live.html").read_text(encoding="utf-8")
    javascript = (live.LIVE_ASSETS_DIR / "live.js").read_text(encoding="utf-8")
    theme = (live.LIVE_ASSETS_DIR / "theme.css").read_text(encoding="utf-8")

    assert "麥克風收音" in html
    assert "分頁／視窗／螢幕收音" in html
    assert "會議混合收音" in html
    assert 'id="mic-button" class="primary" disabled' in html
    assert 'id="screen-button" class="primary" disabled' in html
    assert 'id="mixed-button" class="primary" disabled' in html
    assert "下載 Markdown" in html
    assert "getUserMedia" in javascript
    assert "getDisplayMedia" in javascript
    assert "createMediaStreamDestination" in javascript
    assert 'startRecording("mixed")' in javascript
    assert "function hasProviderKey()" in javascript
    assert "function handleProviderChange()" in javascript
    assert "!state.configLoaded || !hasProviderKey()" in javascript
    assert 'fetch("/api/live/transcribe"' in javascript
    assert "indexedDB.open" in javascript
    assert "MAX_AUTO_ATTEMPTS = 3" in javascript
    assert "function retryChunk" in javascript
    assert 'id="recording-count"' in html
    assert 'id="waiting-count"' in html
    assert 'id="completed-count"' in html
    assert "逐字稿會自動保存在此瀏覽器" in html
    assert 'href="/live-assets/theme.css"' in html
    assert "app-topbar-inner" in html
    assert html.count('class="app-card') == 3
    assert "#e87f24" in theme
    assert "#ffc81e" in theme
    assert "#fefddf" in theme
    assert "#73a5ca" in theme
