from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from .media import media_duration
from .models import TranscriptResult, TranscriptSegment


LANGUAGE_CODES = {
    "自動偵測": None,
    "繁體中文": "zh",
    "英文": "en",
    "日文": "ja",
}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _extract_segments(
    response: Any,
    model: str,
    chunk_duration: float,
) -> list[TranscriptSegment]:
    raw_segments = _value(response, "segments", None)
    if raw_segments:
        segments: list[TranscriptSegment] = []
        for item in raw_segments:
            text = str(_value(item, "text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=max(0.0, float(_value(item, "start", 0.0))),
                    end=max(0.0, float(_value(item, "end", chunk_duration))),
                    text=text,
                    speaker=_value(item, "speaker", None),
                )
            )
        if segments:
            return segments

    text = str(_value(response, "text", response)).strip()
    return [
        TranscriptSegment(
            start=0.0,
            end=max(chunk_duration, 0.1),
            text=text,
        )
    ]


def transcribe_chunks(
    chunks: list[Path],
    *,
    api_key: str,
    model: str,
    language_label: str,
    glossary: str,
    source_name: str,
    progress_callback=None,
) -> TranscriptResult:
    client = OpenAI(api_key=api_key)
    language = LANGUAGE_CODES.get(language_label)
    all_segments: list[TranscriptSegment] = []
    offset = 0.0
    previous_context = ""

    for index, chunk in enumerate(chunks, start=1):
        duration = media_duration(chunk)
        request: dict[str, Any] = {"model": model}
        if language:
            request["language"] = language

        if model == "gpt-4o-transcribe-diarize":
            request["response_format"] = "diarized_json"
            request["chunking_strategy"] = "auto"
        elif model == "whisper-1":
            request["response_format"] = "verbose_json"
            request["timestamp_granularities"] = ["segment"]
            prompt_parts = [glossary.strip(), previous_context[-1200:]]
            prompt = "\n".join(part for part in prompt_parts if part)
            if prompt:
                request["prompt"] = prompt
        else:
            request["response_format"] = "json"
            prompt_parts = [
                "請忠實轉錄，中文使用繁體中文並保留原意與標點。",
                f"專有名詞：{glossary.strip()}" if glossary.strip() else "",
                f"前一段結尾：{previous_context[-1200:]}" if previous_context else "",
            ]
            request["prompt"] = "\n".join(part for part in prompt_parts if part)

        if progress_callback:
            progress_callback(index, len(chunks), chunk.name)

        with chunk.open("rb") as audio_file:
            response = client.audio.transcriptions.create(file=audio_file, **request)

        local_segments = _extract_segments(response, model, duration)
        for segment in local_segments:
            all_segments.append(
                segment.model_copy(
                    update={
                        "start": segment.start + offset,
                        "end": segment.end + offset,
                    }
                )
            )
        previous_context = "\n".join(segment.text for segment in local_segments)
        offset += duration

    return TranscriptResult(
        source_name=source_name,
        language=language,
        model=model,
        segments=all_segments,
    )

