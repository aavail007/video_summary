from __future__ import annotations

import json
from pathlib import Path

from .models import SlideInsight, SummaryResult, TranscriptResult


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcript_markdown(transcript: TranscriptResult) -> str:
    lines = [f"# {transcript.source_name}", "", f"轉錄模型：`{transcript.model}`", ""]
    for segment in transcript.segments:
        speaker = f" **{segment.speaker}**" if segment.speaker else ""
        lines.append(
            f"`{format_timestamp(segment.start)}`{speaker}　{segment.text.strip()}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def summary_markdown(
    summary: SummaryResult,
    slides: list[SlideInsight] | None = None,
) -> str:
    lines = [f"# {summary.title}", "", "## 摘要", "", summary.overview.strip(), ""]
    if summary.key_points:
        lines.extend(["## 重點", ""])
        lines.extend(f"- {item}" for item in summary.key_points)
        lines.append("")
    if summary.chapters:
        lines.extend(["## 章節", ""])
        lines.extend(
            f"- `{item.start_time}` **{item.title}**：{item.summary}"
            for item in summary.chapters
        )
        lines.append("")
    if summary.decisions:
        lines.extend(["## 決策", ""])
        lines.extend(f"- {item}" for item in summary.decisions)
        lines.append("")
    if summary.action_items:
        lines.extend(["## 行動項目", ""])
        for item in summary.action_items:
            details = [value for value in [item.owner, item.deadline] if value]
            suffix = f"（{'／'.join(details)}）" if details else ""
            lines.append(f"- {item.task}{suffix}")
        lines.append("")
    if summary.open_questions:
        lines.extend(["## 未決問題", ""])
        lines.extend(f"- {item}" for item in summary.open_questions)
        lines.append("")
    if slides:
        lines.extend(["## 投影片內容", ""])
        for slide in slides:
            lines.extend(
                [
                    f"### 投影片 {slide.index}｜{format_timestamp(slide.timestamp)}",
                    "",
                    f"![投影片 {slide.index}]({slide.image_file})",
                    "",
                ]
            )
            if slide.title:
                lines.extend([f"**{slide.title}**", ""])
            if slide.visible_text:
                lines.extend(f"- {item}" for item in slide.visible_text)
                lines.append("")
            if slide.visual_summary:
                lines.extend([slide.visual_summary, ""])
    return "\n".join(lines).strip() + "\n"


def write_outputs(
    output_dir: Path,
    transcript: TranscriptResult,
    summary: SummaryResult,
    slides: list[SlideInsight] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "transcript.md": transcript_markdown(transcript),
        "transcript.txt": transcript.text + "\n",
        "transcript.json": json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2),
        "summary.md": summary_markdown(summary, slides),
        "summary.json": json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
    }
    if slides:
        files["slides.json"] = json.dumps(
            [slide.model_dump() for slide in slides],
            ensure_ascii=False,
            indent=2,
        )
    paths: list[Path] = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    if slides:
        paths.extend(
            output_dir / slide.image_file
            for slide in slides
            if (output_dir / slide.image_file).is_file()
        )
    return paths
