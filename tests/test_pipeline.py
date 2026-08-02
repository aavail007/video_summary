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
    available_slide_source_options,
    normalize_slide_source_mode,
    run_pipeline,
)


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
