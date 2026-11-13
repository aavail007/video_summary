import json
from pathlib import Path

import pytest

from video_summary.config import Settings
from video_summary.jobs import incomplete_job_choices, read_job_info, resolve_job_directory
from video_summary.models import TranscriptResult, TranscriptSegment


JOB_ID = "20261010-205946-4b960c2c"


@pytest.fixture
def saved_job(tmp_path):
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", jobs_dir=tmp_path / "data" / "jobs")
    directory = settings.jobs_dir / JOB_ID
    (directory / "chunks").mkdir(parents=True)
    (directory / "cache" / "transcript_chunks").mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({
        "source_name": "測試課程",
        "status": "failed",
    }), encoding="utf-8")
    for index in range(11):
        (directory / "chunks" / f"chunk_{index:03d}.mp3").write_bytes(b"audio")
    for index in range(1, 8):
        transcript = TranscriptResult(
            source_name="測試課程", model="gemini-test", language="zh",
            segments=[TranscriptSegment(start=0, end=1, text="已保存")],
        )
        (directory / "cache" / "transcript_chunks" / f"chunk_{index:03d}.json").write_text(
            transcript.model_dump_json(), encoding="utf-8",
        )
    return settings, directory


def test_job_id_relative_and_absolute_paths_resolve_same_job(saved_job):
    settings, directory = saved_job
    for reference in [JOB_ID, f"data\\jobs\\{JOB_ID}", str(directory), f'"{directory}"']:
        assert resolve_job_directory(settings, reference) == directory.resolve()


def test_job_preview_counts_saved_chunks_and_infers_legacy_settings(saved_job):
    settings, directory = saved_job
    before = (directory / "job.json").read_bytes()
    info = read_job_info(settings, JOB_ID)
    assert info["completed_chunks"] == 7
    assert info["total_chunks"] == 11
    assert info["options"] == {"provider": "Gemini", "gemini_model": "gemini-test", "language_label": "繁體中文"}
    assert not info["transcript_ready"]
    assert incomplete_job_choices(settings)[0][1] == JOB_ID
    assert "7/11" in incomplete_job_choices(settings)[0][0]
    assert (directory / "job.json").read_bytes() == before


def test_completed_and_corrupt_jobs_are_not_offered(saved_job):
    settings, directory = saved_job
    (directory / "job.json").write_text('{"status": "completed"}', encoding="utf-8")
    assert incomplete_job_choices(settings) == []
    (directory / "job.json").write_text("broken json", encoding="utf-8")
    assert incomplete_job_choices(settings) == []


def test_job_path_must_belong_to_configured_jobs_root(saved_job, tmp_path):
    settings, directory = saved_job
    outside = tmp_path / "other" / JOB_ID
    outside.mkdir(parents=True)
    (outside / "job.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="data/jobs"):
        resolve_job_directory(settings, str(outside))
    with pytest.raises(ValueError):
        resolve_job_directory(settings, f"data/jobs/../../other/{JOB_ID}")
