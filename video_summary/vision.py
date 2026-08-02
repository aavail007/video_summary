from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

from .gemini_retry import call_gemini_with_model_fallback
from .models import SlideInsight
from .slides import DetectedSlide


class SlideVisionItem(BaseModel):
    index: int = Field(ge=1)
    content_type: Literal[
        "presentation_slide",
        "played_video",
        "speaker_camera",
        "desktop_or_app",
        "transition_or_other",
    ]
    title: str = ""
    visible_text: list[str] = Field(default_factory=list)
    visual_summary: str = ""


class SlideVisionBatch(BaseModel):
    slides: list[SlideVisionItem]


VISION_PROMPT = """
你是繁體中文簡報閱讀助手。依照每張圖片前的投影片編號分析簡報畫面。
只能描述圖片中可見的資訊，不要從常識補充未出現的事實。
content_type 必須從下列類型中選擇：
- presentation_slide：真正的 PPT／Keynote 簡報頁面，包含標題頁、條列、表格、圖表、流程圖或圖像型投影片。
- played_video：講者在課程中播放的影片、電影、新聞、廣告、示範錄影或其字幕畫面。
- speaker_camera：講者、會議或現場攝影畫面。
- desktop_or_app：桌面、網頁、軟體操作或程式示範，不是簡報頁。
- transition_or_other：轉場、黑畫面、模糊畫面或其他內容。
只有 presentation_slide 會被保留。即使播放影片畫面含有字幕、標題、
圖表或四周有少量 PPT 邊框，只要主要內容是正在播放的影片，必須分類為 played_video。
title：投影片標題；沒有明確標題時用簡短描述。
visible_text：忠實抄錄重要文字、數字、表格欄位與圖例，使用繁體中文。
visual_summary：說明圖表、流程、圖片與各元素之間的關係；純文字頁則簡述版面主旨。
必須為每個輸入編號各回傳一筆，index 必須與輸入相同。
""".strip()


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _image_mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


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
                    "mime_type": _image_mime_type(slide.path),
                },
            ]
        )
    interaction = call_gemini_with_model_fallback(
        lambda active_model: client.interactions.create(
            model=active_model,
            input=inputs,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SlideVisionBatch.model_json_schema(),
            },
        ),
        primary_model=model,
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
                        f"data:{_image_mime_type(slide.path)};base64,"
                        f"{_image_base64(slide.path)}"
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
    excluded_count = 0
    for slide in slides:
        item = analyses.get(slide.index)
        if item is None or item.content_type != "presentation_slide":
            slide.path.unlink(missing_ok=True)
            excluded_count += 1
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
    if progress_callback:
        progress_callback(
            f"投影片篩選完成：保留 {len(results)} 張 PPT，"
            f"排除 {excluded_count} 張播放影片或非簡報畫面"
        )
    return results
