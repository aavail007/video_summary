from pathlib import Path

import pytest

from video_summary.config import Settings
from video_summary.pipeline import PipelineError, _safe_cleanup, _safe_name, run_pipeline


def test_safe_name_removes_path_characters() -> None:
    assert _safe_name('a<b>:"/\\|?*.mp4') == "a_b_.mp4"


def test_cleanup_rejects_outside_job(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PipelineError):
        _safe_cleanup(job, outside)


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
            authorized_content=False,
            api_key_input="test-key",
            transcription_model="gpt-4o-transcribe",
            summary_model="gpt-test",
            gemini_model="gemini-test",
            reasoning_effort="low",
            language_label="自動偵測",
            glossary="",
            summary_style="一般重點摘要",
            analyze_presentation=True,
            delete_temp=True,
        )
