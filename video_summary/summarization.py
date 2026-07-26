from __future__ import annotations

import json

from openai import OpenAI

from .exporters import format_timestamp
from .models import SummaryResult, TranscriptResult


SUMMARY_STYLE_INSTRUCTIONS = {
    "一般重點摘要": "聚焦主旨、重要論點、結論與值得記住的資訊。",
    "會議紀錄": "聚焦討論主題、決策、負責人、待辦事項、期限與未決問題。",
    "課程筆記": "整理概念、定義、例子、推導、章節脈絡與複習重點。",
}


def transcript_for_prompt(transcript: TranscriptResult) -> str:
    lines: list[str] = []
    for segment in transcript.segments:
        timestamp = format_timestamp(segment.start, srt=False)
        speaker = f" {segment.speaker}" if segment.speaker else ""
        lines.append(f"[{timestamp}{speaker}] {segment.text.strip()}")
    return "\n".join(lines)


def _summary_call(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: str,
    content: str,
    style: str,
    partial: bool,
) -> SummaryResult:
    scope = "這是長內容的其中一部分；只整理本段可證實的資訊。" if partial else "整理完整內容。"
    instructions = (
        "你是忠實的繁體中文內容編輯。只能根據逐字稿整理，不可補充逐字稿沒有的事實。"
        "缺少負責人或期限時使用 null。章節 start_time 必須引用逐字稿中已出現的時間。"
        f"{SUMMARY_STYLE_INSTRUCTIONS.get(style, SUMMARY_STYLE_INSTRUCTIONS['一般重點摘要'])}{scope}"
    )
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        instructions=instructions,
        input=content,
        text_format=SummaryResult,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("摘要模型沒有回傳可解析的結構化結果。")
    return parsed


def summarize_transcript(
    transcript: TranscriptResult,
    *,
    api_key: str,
    model: str,
    reasoning_effort: str,
    style: str,
    max_batch_chars: int = 100_000,
) -> SummaryResult:
    client = OpenAI(api_key=api_key)
    content = transcript_for_prompt(transcript)
    if len(content) <= max_batch_chars:
        return _summary_call(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            content=content,
            style=style,
            partial=False,
        )

    batches = [
        content[start : start + max_batch_chars]
        for start in range(0, len(content), max_batch_chars)
    ]
    partials = [
        _summary_call(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            content=batch,
            style=style,
            partial=True,
        )
        for batch in batches
    ]
    combined = "\n\n".join(
        f"第 {index} 部分摘要：\n{json.dumps(item.model_dump(), ensure_ascii=False)}"
        for index, item in enumerate(partials, start=1)
    )
    return _summary_call(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        content=combined,
        style=style,
        partial=False,
    )

