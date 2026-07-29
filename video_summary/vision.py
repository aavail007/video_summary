from __future__ import annotations

import base64
from pathlib import Path

from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

from .gemini_retry import call_gemini_with_retry
from .models import SlideInsight
from .slides import DetectedSlide


class SlideVisionItem(BaseModel):
    index: int = Field(ge=1)
    is_presentation_slide: bool
    title: str = ""
    visible_text: list[str] = Field(default_factory=list)
    visual_summary: str = ""


class SlideVisionBatch(BaseModel):
    slides: list[SlideVisionItem]


VISION_PROMPT = """
你是繁體中文簡報閱讀助手。依照每張圖片前的投影片編號分析簡報畫面。
只能描述圖片中可見的資訊，不要從常識補充未出現的事實。
is_presentation_slide：畫面主要內容是簡報、文件、白板或教學圖表時為 true；
只有講者、攝影畫面、桌面或轉場畫面時為 false。
title：投影片標題；沒有明確標題時用簡短描述。
visible_text：忠實抄錄重要文字、數字、表格欄位與圖例，使用繁體中文。
visual_summary：說明圖表、流程、圖片與各元素之間的關係；純文字頁則簡述版面主旨。
必須為每個輸入編號各回傳一筆，index 必須與輸入相同。
""".strip()


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _slide_batches(
    slides: list[DetectedSlide],
    *,
    maximum_images: int = 10,
    maximum_bytes: int = 12 * 1024 * 1024,
) -> list[list[DetectedSlide]]:
    batches: list[list[DetectedSlide]] = []
    current: list[DetectedSlide] = []
    current_bytes = 0
    for slide in slides:
        size = slide.path.stat().st_size
        if current and (
            len(current) >= maximum_images or current_bytes + size > maximum_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(slide)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def _analyze_gemini_batch(
    client,
    *,
    model: str,
    slides: list[DetectedSlide],
    progress_callback=None,
) -> SlideVisionBatch:
    inputs: list[dict[str, str]] = [{"type": "text", "text": VISION_PROMPT}]
    for slide in slides:
        inputs.extend(
            [
                {
                    "type": "text",
                    "text": f"投影片 index={slide.index}",
                },
                {
                    "type": "image",
                    "data": _image_base64(slide.path),
                    "mime_type": "image/jpeg",
                },
            ]
        )
    interaction = call_gemini_with_retry(
        lambda: client.interactions.create(
            model=model,
            input=inputs,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SlideVisionBatch.model_json_schema(),
            },
        ),
        progress_callback=progress_callback,
    )
    if not interaction.output_text:
        raise RuntimeError("Gemini 沒有回傳投影片分析結果。")
    return SlideVisionBatch.model_validate_json(interaction.output_text)


def _analyze_openai_batch(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: str,
    slides: list[DetectedSlide],
) -> SlideVisionBatch:
    content: list[dict[str, str]] = [
        {"type": "input_text", "text": VISION_PROMPT}
    ]
    for slide in slides:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": f"投影片 index={slide.index}",
                },
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:image/jpeg;base64,{_image_base64(slide.path)}"
                    ),
                    "detail": "high",
                },
            ]
        )
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[{"role": "user", "content": content}],
        text_format=SlideVisionBatch,
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI 沒有回傳投影片分析結果。")
    return response.output_parsed


def analyze_slides(
    slides: list[DetectedSlide],
    *,
    provider: str,
    api_key: str,
    gemini_model: str,
    openai_model: str,
    reasoning_effort: str,
    progress_callback=None,
) -> list[SlideInsight]:
    if not slides:
        return []

    provider_key = provider.strip().lower()
    gemini_client = genai.Client(api_key=api_key) if provider_key == "gemini" else None
    openai_client = OpenAI(api_key=api_key) if provider_key == "openai" else None
    batches = _slide_batches(slides)
    analyses: dict[int, SlideVisionItem] = {}

    for batch_index, batch in enumerate(batches, start=1):
        if progress_callback:
            progress_callback(f"理解投影片內容 {batch_index}/{len(batches)}")
        if provider_key == "gemini":
            parsed = _analyze_gemini_batch(
                gemini_client,
                model=gemini_model,
                slides=batch,
                progress_callback=progress_callback,
            )
        elif provider_key == "openai":
            parsed = _analyze_openai_batch(
                openai_client,
                model=openai_model,
                reasoning_effort=reasoning_effort,
                slides=batch,
            )
        else:
            raise RuntimeError("不支援的投影片分析服務商。")
        analyses.update({item.index: item for item in parsed.slides})

    results: list[SlideInsight] = []
    for slide in slides:
        item = analyses.get(slide.index)
        if item is not None and not item.is_presentation_slide:
            slide.path.unlink(missing_ok=True)
            continue
        results.append(
            SlideInsight(
                index=slide.index,
                timestamp=slide.timestamp,
                image_file=f"slides/{slide.path.name}",
                title=item.title if item else "",
                visible_text=item.visible_text if item else [],
                visual_summary=item.visual_summary if item else "",
            )
        )
    return results
