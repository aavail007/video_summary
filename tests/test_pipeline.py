from pathlib import Path

import pytest

from video_summary.config import Settings
from video_summary.pipeline import (
    SLIDE_SOURCE_AUTO,
    SLIDE_SOURCE_EXISTING,
    SLIDE_SOURCE_NONE,
    SLIDE_SOURCE_OPTIONS,
    PipelineError,
    _safe_cleanup,
    _safe_name,
    _transcribe_chunks_with_cache,
    _write_json,
    available_slide_source_options,
    normalize_slide_source_mode,
    run_pipeline,
)
from video_summary.models import SummaryResult, TranscriptResult, TranscriptSegment


def test_safe_name_removes_path_characters() -> None:
    assert _safe_name('a<b>:"/\\|?*.mp4') == "a_b_.mp4"


def test_cleanup_rejects_outside_job(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PipelineError):
        _safe_cleanup(job, outside)


def test_youtube_url_disables_automatic_slide_extraction() -> None:
    options = available_slide_source_options(
        None,
        "https://www.youtube.com/watch?v=test",
    )

    assert options == [SLIDE_SOURCE_NONE, SLIDE_SOURCE_EXISTING]
    assert (
        normalize_slide_source_mode(None, "https://youtu.be/test", SLIDE_SOURCE_AUTO)
        == SLIDE_SOURCE_NONE
    )
    assert (
        normalize_slide_source_mode(
            None,
            "https://youtu.be/test",
            SLIDE_SOURCE_EXISTING,
        )
        == SLIDE_SOURCE_EXISTING
    )


def test_empty_youtube_value_accepts_none() -> None:
    assert SLIDE_SOURCE_AUTO in available_slide_source_options(None, None)


def test_slide_source_order_starts_with_audio_only() -> None:
    assert SLIDE_SOURCE_OPTIONS == [
        SLIDE_SOURCE_NONE,
        SLIDE_SOURCE_EXISTING,
        SLIDE_SOURCE_AUTO,
    ]


def test_audio_upload_disables_automatic_slide_extraction() -> None:
    assert available_slide_source_options("lecture.mp3", "") == [
        SLIDE_SOURCE_NONE,
        SLIDE_SOURCE_EXISTING,
    ]
    assert (
        normalize_slide_source_mode("lecture.mp3", "", SLIDE_SOURCE_AUTO)
        == SLIDE_SOURCE_NONE
    )


def test_video_upload_keeps_all_slide_sources() -> None:
    assert SLIDE_SOURCE_AUTO in available_slide_source_options("lecture.mp4", "")
    assert (
        normalize_slide_source_mode("lecture.mp4", "", SLIDE_SOURCE_AUTO)
        == SLIDE_SOURCE_AUTO
    )


def test_pipeline_rejects_unknown_provider(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="AI 服務商"):
        run_pipeline(
            settings=Settings(
                project_root=tmp_path,
                data_dir=tmp_path / "data",
                jobs_dir=tmp_path / "data" / "jobs",
            ),
            provider="Unknown",
            uploaded_path=None,
            youtube_url="",
            api_key_input="test-key",
            transcription_model="gpt-4o-transcribe",
            summary_model="gpt-test",
            gemini_model="gemini-test",
            reasoning_effort="low",
            language_label="自動偵測",
            glossary="",
            summary_style="一般重點摘要",
            slide_source_mode="自動從影片偵測並擷取",
            existing_slide_paths=[],
            delete_temp=True,
        )


def test_chunk_cache_retries_only_missing_transcription(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_summary import pipeline

    chunks = []
    for index in range(3):
        chunk = tmp_path / f"chunk_{index:03d}.mp3"
        chunk.write_bytes(b"audio")
        chunks.append(chunk)

    calls: list[str] = []
    fail_second = True

    def fake_transcribe(selected, **kwargs):
        nonlocal fail_second
        name = selected[0].name
        calls.append(name)
        if name == "chunk_001.mp3" and fail_second:
            fail_second = False
            raise RuntimeError("temporary failure")
        return TranscriptResult(
            source_name="lecture",
            language="zh",
            model="test-model",
            segments=[TranscriptSegment(start=0, end=3, text=name)],
        )

    monkeypatch.setattr(pipeline, "transcribe_chunks", fake_transcribe)
    monkeypatch.setattr(pipeline, "media_duration", lambda path: 10.0)
    cache_dir = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="temporary failure"):
        _transcribe_chunks_with_cache(
            chunks,
            cache_dir=cache_dir,
            provider_key="openai",
            api_key="key",
            transcription_model="test-model",
            gemini_model="gemini-test",
            language_label="繁體中文",
            glossary="",
            source_name="lecture",
        )

    result = _transcribe_chunks_with_cache(
        chunks,
        cache_dir=cache_dir,
        provider_key="openai",
        api_key="key",
        transcription_model="test-model",
        gemini_model="gemini-test",
        language_label="繁體中文",
        glossary="",
        source_name="lecture",
    )

    assert calls.count("chunk_000.mp3") == 1
    assert calls.count("chunk_001.mp3") == 2
    assert calls.count("chunk_002.mp3") == 1
    assert [segment.start for segment in result.segments] == [0, 10, 20]


def test_summary_only_uses_cached_transcript_without_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_summary import pipeline

    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        jobs_dir=tmp_path / "data" / "jobs",
        openai_api_key="configured-key",
    )
    job_id = "20260812-120000-a1b2c3d4"
    job_dir = settings.jobs_dir / job_id
    _write_json(
        job_dir / "job.json",
        {
            "version": 1,
            "job_id": job_id,
            "slide_source_mode": SLIDE_SOURCE_NONE,
            "source_path": "",
            "source_name": "lecture",
        },
    )
    _write_json(
        job_dir / "cache" / "transcript.json",
        TranscriptResult(
            source_name="lecture",
            language="zh",
            model="test-transcribe",
            segments=[TranscriptSegment(start=0, end=2, text="cached words")],
        ).model_dump(),
    )
    summary_calls: list[str] = []

    def fake_summary(transcript, **kwargs):
        summary_calls.append(transcript.text)
        return SummaryResult(title="New summary", overview="Done")

    monkeypatch.setattr(pipeline, "summarize_transcript", fake_summary)

    transcript_md, summary_md, files, status, returned_job_id = run_pipeline(
        settings=settings,
        provider="OpenAI",
        uploaded_path=None,
        youtube_url="",
        api_key_input="",
        transcription_model="test-transcribe",
        summary_model="test-summary",
        gemini_model="gemini-test",
        reasoning_effort="low",
        language_label="繁體中文",
        glossary="",
        summary_style="一般重點摘要",
        slide_source_mode=SLIDE_SOURCE_NONE,
        existing_slide_paths=[],
        delete_temp=True,
        resume_job_id=job_id,
        summary_only=True,
    )

    assert summary_calls == ["cached words"]
    assert "cached words" in transcript_md
    assert "New summary" in summary_md
    assert returned_job_id == job_id
    assert "重新摘要完成" in status
    assert files


def test_resume_after_refresh_uses_saved_audio_and_skips_seven_completed_chunks(tmp_path, monkeypatch):
    from video_summary import pipeline

    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", jobs_dir=tmp_path / "data" / "jobs")
    job_id = "20261010-205946-4b960c2c"
    job_dir = settings.jobs_dir / job_id
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    _write_json(job_dir / "job.json", {
        "status": "failed", "source_name": "lecture", "source_path": "",
        "slide_source_mode": SLIDE_SOURCE_NONE,
    })
    for index in range(11):
        (chunks_dir / f"chunk_{index:03d}.mp3").write_bytes(b"audio")
    for index in range(1, 8):
        _write_json(job_dir / "cache" / "transcript_chunks" / f"chunk_{index:03d}.json", TranscriptResult(
            source_name="lecture", model="gemini-test", language="zh",
            segments=[TranscriptSegment(start=0, end=2, text=f"saved-{index}")],
        ).model_dump())

    calls = []

    def transcribe(chunks, **kwargs):
        calls.append(chunks[0].name)
        return TranscriptResult(source_name="lecture", model="gemini-test", language="zh",
                                segments=[TranscriptSegment(start=0, end=2, text="new words")])

    def unexpected(*args, **kwargs):
        pytest.fail("續跑不應重新下載、複製或切割來源")

    monkeypatch.setattr(pipeline, "transcribe_chunks_gemini", transcribe)
    monkeypatch.setattr(pipeline, "media_duration", lambda path: 10.0)
    monkeypatch.setattr(pipeline, "split_audio", unexpected)
    monkeypatch.setattr(pipeline, "_copy_upload", unexpected)
    monkeypatch.setattr(pipeline, "download_authorized_audio", unexpected)
    monkeypatch.setattr(pipeline, "summarize_transcript_gemini",
                        lambda *args, **kwargs: SummaryResult(title="summary", overview="done"))
    result = run_pipeline(
        settings=settings, provider="Gemini", uploaded_path=None, youtube_url="", api_key_input="test-secret",
        transcription_model=None, summary_model=None, gemini_model="gemini-test", reasoning_effort="low",
        language_label="繁體中文", glossary="", summary_style="課程筆記", slide_source_mode=SLIDE_SOURCE_NONE,
        existing_slide_paths=[], delete_temp=False, resume_job_id=f"data\\jobs\\{job_id}",
    )
    assert calls == [f"chunk_{index:03d}.mp3" for index in range(7, 11)]
    assert result[4] == job_id
    assert "saved-1" in result[0]
    manifest = (job_dir / "job.json").read_text(encoding="utf-8")
    assert "test-secret" not in manifest
    assert '"status": "completed"' in manifest
