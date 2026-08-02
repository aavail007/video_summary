from pathlib import Path

from video_summary.exporters import (
    format_timestamp,
    summary_markdown,
    transcript_markdown,
    write_outputs,
)
from video_summary.models import (
    ActionItem,
    Chapter,
    SlideInsight,
    SummaryResult,
    TranscriptResult,
    TranscriptSegment,
)


def sample_transcript() -> TranscriptResult:
    return TranscriptResult(
        source_name="測試影片",
        language="zh",
        model="whisper-1",
        segments=[
            TranscriptSegment(start=0, end=2.5, text="大家好。", speaker="A"),
            TranscriptSegment(start=2.5, end=5, text="今天介紹測試。", speaker="B"),
        ],
    )


def test_timestamp_formats() -> None:
    assert format_timestamp(3661.25) == "01:01:01.250"


def test_transcript_exports() -> None:
    transcript = sample_transcript()
    markdown = transcript_markdown(transcript)
    assert "# 測試影片" in markdown
    assert "**A**" in markdown


def test_write_outputs_does_not_create_srt(tmp_path: Path) -> None:
    paths = write_outputs(
        tmp_path,
        sample_transcript(),
        SummaryResult(title="摘要", overview="內容"),
    )

    assert not (tmp_path / "transcript.srt").exists()
    assert "transcript.srt" not in {path.name for path in paths}


def test_summary_markdown() -> None:
    summary = SummaryResult(
        title="摘要標題",
        overview="摘要內容",
        key_points=["重點一"],
        chapters=[Chapter(start_time="00:00:00", title="開場", summary="介紹")],
        decisions=["採用方案 A"],
        action_items=[ActionItem(task="完成測試", owner="小明", deadline=None)],
        open_questions=["何時上線？"],
    )
    markdown = summary_markdown(
        summary,
        slides=[
            SlideInsight(
                index=1,
                timestamp=12.5,
                image_file="slides/slide_001.jpg",
                title="測試投影片",
                visible_text=["畫面文字"],
                visual_summary="圖表說明",
            )
        ],
    )
    assert "# 摘要標題" in markdown
    assert "採用方案 A" in markdown
    assert "完成測試（小明）" in markdown
    assert "投影片 1｜00:00:12.500" in markdown
    assert "slides/slide_001.jpg" in markdown
