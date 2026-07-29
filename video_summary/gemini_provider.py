from __future__ import annotations

import base64
import json
from pathlib import Path

from google import genai
from pydantic import BaseModel, Field

from .gemini_retry import call_gemini_with_retry
from .media import media_duration
from .models import SlideInsight, SummaryResult, TranscriptResult, TranscriptSegment
from .summarization import SUMMARY_STYLE_INSTRUCTIONS, content_for_summary
from .transcription import LANGUAGE_CODES


class GeminiTranscriptSegment(BaseModel):
    start_seconds: float = Field(
        ge=0,
        description="Segment start time in seconds from the beginning of this audio file.",
    )
    end_seconds: float = Field(
        ge=0,
        description="Segment end time in seconds from the beginning of this audio file.",
    )
    text: str = Field(description="Verbatim transcript in the requested writing system.")
    speaker: str | None = Field(
        default=None,
        description="Consistent speaker label, such as Speaker 1, when distinguishable.",
    )


class GeminiTranscriptResponse(BaseModel):
    detected_language: str | None = None
    segments: list[GeminiTranscriptSegment]


def _audio_input(path: Path) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "audio",
        "data": encoded,
        "mime_type": "audio/mp3",
    }


def _interaction_json(
    client,
    *,
    model: str,
    inputs,
    schema: type[BaseModel],
    retry_callback=None,
) -> BaseModel:
    interaction = call_gemini_with_retry(
        lambda: client.interactions.create(
            model=model,
            input=inputs,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
        ),
        progress_callback=retry_callback,
    )
    output_text = getattr(interaction, "output_text", "")
    if not output_text:
        raise RuntimeError("Gemini 沒有回傳可解析的內容。")
    return schema.model_validate_json(output_text)


def transcribe_chunks_gemini(
    chunks: list[Path],
    *,
    api_key: str,
    model: str,
    language_label: str,
    glossary: str,
    source_name: str,
    progress_callback=None,
    retry_callback=None,
) -> TranscriptResult:
    client = genai.Client(api_key=api_key)
    requested_language = LANGUAGE_CODES.get(language_label)
    all_segments: list[TranscriptSegment] = []
    offset = 0.0
    previous_context = ""
    detected_language: str | None = requested_language

    for index, chunk in enumerate(chunks, start=1):
        if chunk.stat().st_size >= 18 * 1024 * 1024:
            raise RuntimeError("Gemini inline 音訊片段必須小於 18 MB。")
        duration = media_duration(chunk)
        language_instruction = (
            "主要語言為繁體中文，請使用繁體中文字形。"
            if language_label == "繁體中文"
            else (
                f"主要語言代碼為 {requested_language}。"
                if requested_language
                else "自動辨識語言；中文內容請使用繁體中文字形。"
            )
        )
        prompt_parts = [
            "忠實轉錄這段音訊，不可摘要、改寫或補充沒有說出的內容。",
            language_instruction,
            "辨識不同講者，為每段提供相對於這個音訊片段的開始與結束秒數。",
            "聽不清楚時保留最接近的原話，不要自行猜測具體事實。",
            f"專有名詞：{glossary.strip()}" if glossary.strip() else "",
            f"前一段結尾：{previous_context[-1200:]}" if previous_context else "",
        ]
        prompt = "\n".join(part for part in prompt_parts if part)

        if progress_callback:
            progress_callback(index, len(chunks), chunk.name)

        parsed = _interaction_json(
            client,
            model=model,
            inputs=[
                {"type": "text", "text": prompt},
                _audio_input(chunk),
            ],
            schema=GeminiTranscriptResponse,
            retry_callback=retry_callback,
        )
        assert isinstance(parsed, GeminiTranscriptResponse)
        detected_language = parsed.detected_language or detected_language
        for segment in parsed.segments:
            text = segment.text.strip()
            if not text:
                continue
            end = max(segment.end_seconds, segment.start_seconds + 0.1)
            all_segments.append(
                TranscriptSegment(
                    start=offset + segment.start_seconds,
                    end=offset + min(end, duration or end),
                    text=text,
                    speaker=segment.speaker,
                )
            )
        previous_context = "\n".join(
            segment.text.strip() for segment in parsed.segments if segment.text.strip()
        )
        offset += duration

    if not all_segments:
        raise RuntimeError("Gemini 沒有辨識出可用的逐字稿。")
    return TranscriptResult(
        source_name=source_name,
        language=detected_language,
        model=model,
        segments=all_segments,
    )


def _gemini_summary_call(
    client,
    *,
    model: str,
    content: str,
    style: str,
    partial: bool,
    retry_callback=None,
) -> SummaryResult:
    scope = "這是長內容的其中一部分；只整理本段可證實的資訊。" if partial else "整理完整內容。"
    prompt = (
        "你是忠實的繁體中文內容編輯。只能根據逐字稿整理，不可補充逐字稿沒有的事實。"
        "缺少負責人或期限時使用 null。章節 start_time 必須引用逐字稿中已出現的時間。"
        f"{SUMMARY_STYLE_INSTRUCTIONS.get(style, SUMMARY_STYLE_INSTRUCTIONS['一般重點摘要'])}"
        f"{scope}\n\n內容資料：\n{content}"
    )
    parsed = _interaction_json(
        client,
        model=model,
        inputs=prompt,
        schema=SummaryResult,
        retry_callback=retry_callback,
    )
    assert isinstance(parsed, SummaryResult)
    return parsed


def summarize_transcript_gemini(
    transcript: TranscriptResult,
    *,
    api_key: str,
    model: str,
    style: str,
    slides: list[SlideInsight] | None = None,
    max_batch_chars: int = 100_000,
    progress_callback=None,
) -> SummaryResult:
    client = genai.Client(api_key=api_key)
    content = content_for_summary(transcript, slides)
    if len(content) <= max_batch_chars:
        return _gemini_summary_call(
            client,
            model=model,
            content=content,
            style=style,
            partial=False,
            retry_callback=progress_callback,
        )

    batches = [
        content[start : start + max_batch_chars]
        for start in range(0, len(content), max_batch_chars)
    ]
    partials = [
        _gemini_summary_call(
            client,
            model=model,
            content=batch,
            style=style,
            partial=True,
            retry_callback=progress_callback,
        )
        for batch in batches
    ]
    combined = "\n\n".join(
        f"第 {index} 部分摘要：\n{json.dumps(item.model_dump(), ensure_ascii=False)}"
        for index, item in enumerate(partials, start=1)
    )
    return _gemini_summary_call(
        client,
        model=model,
        content=combined,
        style=style,
        partial=False,
        retry_callback=progress_callback,
    )
