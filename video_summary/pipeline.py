from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .config import Settings
from .exporters import summary_markdown, transcript_markdown, write_outputs
from .gemini_provider import summarize_transcript_gemini, transcribe_chunks_gemini
from .media import split_audio
from .slides import extract_unique_slides, import_existing_slides, is_video_file
from .summarization import summarize_transcript
from .transcription import transcribe_chunks
from .vision import analyze_slides
from .youtube import download_authorized_audio


class PipelineError(RuntimeError):
    pass


SLIDE_SOURCE_AUTO = "自動從影片偵測並擷取"
SLIDE_SOURCE_EXISTING = "使用之前已擷取的投影片"
SLIDE_SOURCE_NONE = "不分析投影片，只分析音訊"
SLIDE_SOURCE_OPTIONS = [
    SLIDE_SOURCE_AUTO,
    SLIDE_SOURCE_EXISTING,
    SLIDE_SOURCE_NONE,
]


def available_slide_source_options(
    uploaded_path: str | None,
    youtube_url: str | None,
) -> list[str]:
    """Return slide modes that can work with the currently selected media source."""
    if (youtube_url or "").strip():
        return [SLIDE_SOURCE_EXISTING, SLIDE_SOURCE_NONE]
    if uploaded_path and not is_video_file(Path(uploaded_path)):
        return [SLIDE_SOURCE_EXISTING, SLIDE_SOURCE_NONE]
    return list(SLIDE_SOURCE_OPTIONS)


def normalize_slide_source_mode(
    uploaded_path: str | None,
    youtube_url: str | None,
    selected_mode: str,
) -> str:
    options = available_slide_source_options(uploaded_path, youtube_url)
    return selected_mode if selected_mode in options else SLIDE_SOURCE_NONE


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip(" ._")
    return cleaned[:120] or "media"


def _copy_upload(uploaded_path: str, target_dir: Path) -> tuple[Path, str]:
    source = Path(uploaded_path)
    if not source.is_file():
        raise PipelineError("找不到上傳的檔案。")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _safe_name(source.name)
    shutil.copy2(source, target)
    return target, source.stem


def _safe_cleanup(job_dir: Path, *targets: Path) -> None:
    job_root = job_dir.resolve()
    for target in targets:
        resolved = target.resolve()
        if resolved == job_root or job_root not in resolved.parents:
            raise PipelineError("拒絕清除工作目錄以外的路徑。")
        if resolved.exists():
            shutil.rmtree(resolved)


def run_pipeline(
    *,
    settings: Settings,
    provider: str,
    uploaded_path: str | None,
    youtube_url: str | None,
    api_key_input: str | None,
    transcription_model: str | None,
    summary_model: str | None,
    gemini_model: str | None,
    reasoning_effort: str,
    language_label: str,
    glossary: str,
    summary_style: str,
    slide_source_mode: str,
    existing_slide_paths: list[str] | None,
    delete_temp: bool,
    status_callback=None,
) -> tuple[str, str, list[str], str]:
    provider = (provider or "").strip()
    youtube_url = (youtube_url or "").strip()
    transcription_model = (transcription_model or settings.transcription_model).strip()
    summary_model = (summary_model or settings.summary_model).strip()
    gemini_model = (gemini_model or settings.gemini_model).strip()
    glossary = glossary or ""
    provider_key = provider.lower()
    if provider_key not in {"gemini", "openai"}:
        raise PipelineError("不支援的 AI 服務商。")
    if slide_source_mode not in SLIDE_SOURCE_OPTIONS:
        raise PipelineError("不支援的投影片來源模式。")
    configured_key = (
        settings.gemini_api_key if provider_key == "gemini" else settings.openai_api_key
    )
    api_key = (api_key_input or "").strip() or configured_key
    if not api_key:
        environment_name = "GEMINI_API_KEY" if provider_key == "gemini" else "OPENAI_API_KEY"
        raise PipelineError(
            f"請在畫面輸入 {provider} API Key，或在 .env 設定 {environment_name}。"
        )
    if not uploaded_path and not youtube_url:
        raise PipelineError("請上傳音檔／影片，或輸入 YouTube 網址。")
    if uploaded_path and youtube_url:
        raise PipelineError("請只選擇一種來源：上傳檔案或 YouTube 網址。")
    slide_source_mode = normalize_slide_source_mode(
        uploaded_path,
        youtube_url,
        slide_source_mode,
    )

    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    job_dir = settings.jobs_dir / job_id
    source_dir = job_dir / "source"
    chunks_dir = job_dir / "chunks"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if status_callback:
        status_callback("準備來源檔案")
    if uploaded_path:
        source_path, source_name = _copy_upload(uploaded_path, source_dir)
    else:
        source_path, source_name = download_authorized_audio(
            youtube_url,
            source_dir,
            settings.ytdlp_cookies_file,
        )

    if status_callback:
        status_callback("在本機抽取並切割音訊")
    detected_slides = []
    if slide_source_mode == SLIDE_SOURCE_AUTO:
        detected_slides = extract_unique_slides(
            source_path,
            output_dir / "slides",
            progress_callback=status_callback,
        )
    elif slide_source_mode == SLIDE_SOURCE_EXISTING:
        detected_slides = import_existing_slides(
            existing_slide_paths or [],
            output_dir / "slides",
            progress_callback=status_callback,
        )
    chunks = split_audio(source_path, chunks_dir, settings.chunk_seconds)

    def on_chunk(index: int, total: int, name: str) -> None:
        if status_callback:
            status_callback(f"轉錄音訊 {index}/{total}：{name}")

    if provider_key == "gemini":
        transcript = transcribe_chunks_gemini(
            chunks,
            api_key=api_key,
            model=gemini_model,
            language_label=language_label,
            glossary=glossary,
            source_name=source_name,
            progress_callback=on_chunk,
            retry_callback=status_callback,
        )
    else:
        transcript = transcribe_chunks(
            chunks,
            api_key=api_key,
            model=transcription_model,
            language_label=language_label,
            glossary=glossary,
            source_name=source_name,
            progress_callback=on_chunk,
        )

    slide_insights = []
    if detected_slides:
        slide_insights = analyze_slides(
            detected_slides,
            provider=provider,
            api_key=api_key,
            gemini_model=gemini_model,
            openai_model=summary_model,
            reasoning_effort=reasoning_effort,
            progress_callback=status_callback,
        )

    if status_callback:
        status_callback(f"使用 {provider} 產生結構化摘要")
    if provider_key == "gemini":
        summary = summarize_transcript_gemini(
            transcript,
            api_key=api_key,
            model=gemini_model,
            style=summary_style,
            slides=slide_insights,
            progress_callback=status_callback,
        )
    else:
        summary = summarize_transcript(
            transcript,
            api_key=api_key,
            model=summary_model,
            reasoning_effort=reasoning_effort,
            style=summary_style,
            slides=slide_insights,
        )

    if status_callback:
        status_callback("寫入本機輸出檔")
    output_paths = write_outputs(
        output_dir,
        transcript,
        summary,
        slides=slide_insights,
    )

    if delete_temp:
        _safe_cleanup(job_dir, source_dir, chunks_dir)

    status = f"完成（{provider}）。結果位於：{output_dir}"
    return (
        transcript_markdown(transcript),
        summary_markdown(summary, slide_insights),
        [str(path) for path in output_paths],
        status,
    )
