from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .config import Settings
from .exporters import summary_markdown, transcript_markdown, write_outputs
from .gemini_provider import summarize_transcript_gemini, transcribe_chunks_gemini
from .jobs import resolve_job_directory
from .media import media_duration, split_audio
from .models import SlideInsight, SummaryResult, TranscriptResult, TranscriptSegment
from .slides import extract_unique_slides, import_existing_slides, is_video_file
from .summarization import summarize_transcript
from .transcription import transcribe_chunks
from .vision import analyze_slides
from .youtube import download_authorized_audio


class PipelineError(RuntimeError):
    def __init__(self, message: str, *, job_id: str = "") -> None:
        super().__init__(message)
        self.job_id = job_id


SLIDE_SOURCE_AUTO = "自動從影片偵測並擷取"
SLIDE_SOURCE_EXISTING = "使用之前已擷取的投影片"
SLIDE_SOURCE_NONE = "不分析投影片，只分析音訊"
SLIDE_SOURCE_OPTIONS = [
    SLIDE_SOURCE_NONE,
    SLIDE_SOURCE_EXISTING,
    SLIDE_SOURCE_AUTO,
]


def available_slide_source_options(
    uploaded_path: str | None,
    youtube_url: str | None,
) -> list[str]:
    """Return slide modes that can work with the currently selected media source."""
    if (youtube_url or "").strip():
        return [SLIDE_SOURCE_NONE, SLIDE_SOURCE_EXISTING]
    if uploaded_path and not is_video_file(Path(uploaded_path)):
        return [SLIDE_SOURCE_NONE, SLIDE_SOURCE_EXISTING]
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


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_job_dir(settings: Settings, job_id: str) -> Path:
    try:
        return resolve_job_directory(settings, job_id)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc


def _load_detected_slides(path: Path):
    from .slides import DetectedSlide

    return [
        DetectedSlide(
            index=int(item["index"]),
            timestamp=float(item["timestamp"]),
            path=Path(item["path"]),
        )
        for item in _read_json(path)
        if Path(item["path"]).is_file()
    ]


def _save_detected_slides(path: Path, slides) -> None:
    _write_json(
        path,
        [
            {
                "index": slide.index,
                "timestamp": slide.timestamp,
                "path": str(slide.path),
            }
            for slide in slides
        ],
    )


def _transcribe_chunks_with_cache(
    chunks: list[Path],
    *,
    cache_dir: Path,
    provider_key: str,
    api_key: str,
    transcription_model: str,
    gemini_model: str,
    language_label: str,
    glossary: str,
    source_name: str,
    status_callback=None,
) -> TranscriptResult:
    """Transcribe one chunk at a time and retain each successful chunk for resume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined: list[TranscriptSegment] = []
    offset = 0.0
    previous_context = ""
    detected_language: str | None = None
    active_model = gemini_model if provider_key == "gemini" else transcription_model

    for index, chunk in enumerate(chunks, start=1):
        cache_path = cache_dir / f"chunk_{index:03d}.json"
        if cache_path.is_file():
            result = TranscriptResult.model_validate(_read_json(cache_path))
            if status_callback:
                status_callback(f"沿用已完成的轉錄片段 {index}/{len(chunks)}")
        else:
            if status_callback:
                status_callback(f"轉錄片段 {index}/{len(chunks)}：{chunk.name}")
            if provider_key == "gemini":
                result = transcribe_chunks_gemini(
                    [chunk],
                    api_key=api_key,
                    model=gemini_model,
                    language_label=language_label,
                    glossary=glossary,
                    source_name=source_name,
                    initial_context=previous_context,
                    retry_callback=status_callback,
                    request_label=f"轉錄片段 {index}/{len(chunks)}",
                )
            else:
                result = transcribe_chunks(
                    [chunk],
                    api_key=api_key,
                    model=transcription_model,
                    language_label=language_label,
                    glossary=glossary,
                    source_name=source_name,
                    initial_context=previous_context,
                )
            _write_json(cache_path, result.model_dump())

        detected_language = result.language or detected_language
        active_model = result.model or active_model
        for segment in result.segments:
            combined.append(
                segment.model_copy(
                    update={
                        "start": segment.start + offset,
                        "end": segment.end + offset,
                    }
                )
            )
        previous_context = result.text
        duration = media_duration(chunk)
        if duration <= 0 and result.segments:
            duration = max(segment.end for segment in result.segments)
        offset += max(duration, 0.0)

    if not combined:
        raise PipelineError("沒有辨識出可用的逐字稿。")
    return TranscriptResult(
        source_name=source_name,
        language=detected_language,
        model=active_model,
        segments=combined,
    )


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
    resume_job_id: str = "",
    summary_only: bool = False,
    status_callback=None,
) -> tuple[str, str, list[str], str, str]:
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
        raise PipelineError("不支援的投影片來源選項。")
    configured_key = (
        settings.gemini_api_key if provider_key == "gemini" else settings.openai_api_key
    )
    api_key = (api_key_input or "").strip() or configured_key
    if not api_key:
        environment_name = "GEMINI_API_KEY" if provider_key == "gemini" else "OPENAI_API_KEY"
        raise PipelineError(
            f"請在畫面輸入 {provider} API Key，或於 .env 設定 {environment_name}。"
        )
    if not resume_job_id and not uploaded_path and not youtube_url:
        raise PipelineError("請上傳音檔或影片，或輸入 YouTube 網址。")
    if not resume_job_id and uploaded_path and youtube_url:
        raise PipelineError("請只選擇一種來源：上傳檔案或 YouTube 網址。")

    if resume_job_id:
        job_dir = _resolve_job_dir(settings, resume_job_id)
        job_id = job_dir.name
    else:
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
    cache_dir = job_dir / "cache"
    manifest_path = job_dir / "job.json"
    transcript_cache = cache_dir / "transcript.json"
    detected_slides_cache = cache_dir / "detected_slides.json"
    slide_insights_cache = cache_dir / "slides.json"
    summary_cache = cache_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            slide_source_mode = manifest.get("slide_source_mode", SLIDE_SOURCE_NONE)
        else:
            manifest = {
                "version": 1,
                "job_id": job_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "slide_source_mode": slide_source_mode,
                "source_path": "",
                "source_name": "",
            }
            _write_json(manifest_path, manifest)

        # Store only non-secret settings so refreshing the UI can restore them.
        manifest["options"] = {
            "provider": provider,
            "transcription_model": transcription_model,
            "summary_model": summary_model,
            "gemini_model": gemini_model,
            "reasoning_effort": reasoning_effort,
            "language_label": language_label,
            "glossary": glossary,
            "summary_style": summary_style,
        }
        _write_json(manifest_path, manifest)

        source_path = Path(manifest["source_path"]) if manifest.get("source_path") else None
        source_name = str(manifest.get("source_name") or "影音內容")
        source_ready = bool(source_path and source_path.is_file())

        saved_chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
        if not source_ready and not saved_chunks and not summary_only and not transcript_cache.is_file():
            if resume_job_id and not uploaded_path and not youtube_url:
                raise PipelineError("原始影音已被刪除，且尚未完成轉錄，無法續跑。")
            if resume_job_id and manifest.get("source_path"):
                raise PipelineError("此工作的原始影音與音訊分段已遺失；請建立新工作，避免混用舊轉錄快取。")
            if status_callback:
                status_callback("準備影音來源")
            if uploaded_path:
                source_path, source_name = _copy_upload(uploaded_path, source_dir)
            else:
                source_path, source_name = download_authorized_audio(
                    youtube_url,
                    source_dir,
                    settings.ytdlp_cookies_file,
                )
            manifest.update(source_path=str(source_path), source_name=source_name)
            _write_json(manifest_path, manifest)
            source_ready = True

        if transcript_cache.is_file():
            transcript = TranscriptResult.model_validate(_read_json(transcript_cache))
            if status_callback:
                status_callback("沿用已完成的逐字稿")
        else:
            if summary_only:
                raise PipelineError("此工作尚未完成逐字稿，不能只重新摘要。")
            if not saved_chunks and (not source_path or not source_ready):
                raise PipelineError("找不到可供轉錄的原始影音。")
            chunks = saved_chunks
            if not chunks:
                if status_callback:
                    status_callback("建立音訊分段")
                chunks = split_audio(source_path, chunks_dir, settings.chunk_seconds)
            transcript = _transcribe_chunks_with_cache(
                chunks,
                cache_dir=cache_dir / "transcript_chunks",
                provider_key=provider_key,
                api_key=api_key,
                transcription_model=transcription_model,
                gemini_model=gemini_model,
                language_label=language_label,
                glossary=glossary,
                source_name=source_name,
                status_callback=status_callback,
            )
            _write_json(transcript_cache, transcript.model_dump())

        slide_insights: list[SlideInsight] = []
        if slide_insights_cache.is_file():
            slide_insights = [
                SlideInsight.model_validate(item)
                for item in _read_json(slide_insights_cache)
            ]
            if status_callback and slide_insights:
                status_callback("沿用已完成的投影片分析")
        elif not summary_only and slide_source_mode != SLIDE_SOURCE_NONE:
            if detected_slides_cache.is_file():
                detected_slides = _load_detected_slides(detected_slides_cache)
            else:
                if not source_path or not source_ready:
                    raise PipelineError("找不到原始影片，無法繼續擷取投影片。")
                if slide_source_mode == SLIDE_SOURCE_AUTO:
                    detected_slides = extract_unique_slides(
                        source_path,
                        output_dir / "slides",
                        progress_callback=status_callback,
                    )
                else:
                    detected_slides = import_existing_slides(
                        existing_slide_paths or [],
                        output_dir / "slides",
                        progress_callback=status_callback,
                    )
                _save_detected_slides(detected_slides_cache, detected_slides)
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
                _write_json(
                    slide_insights_cache,
                    [item.model_dump() for item in slide_insights],
                )

        if summary_cache.is_file() and not summary_only:
            summary = SummaryResult.model_validate(_read_json(summary_cache))
            if status_callback:
                status_callback("沿用已完成的摘要")
        else:
            if status_callback:
                status_callback(f"使用 {provider} 產生重點摘要")
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
            _write_json(summary_cache, summary.model_dump())

        if status_callback:
            status_callback("寫入本機輸出檔案")
        output_paths = write_outputs(
            output_dir,
            transcript,
            summary,
            slides=slide_insights,
        )
        manifest.update(
            completed_at=datetime.now().isoformat(timespec="seconds"),
            status="completed",
            error="",
        )
        _write_json(manifest_path, manifest)

        if delete_temp:
            _safe_cleanup(job_dir, source_dir, chunks_dir)

        action_label = "重新摘要完成" if summary_only else "完成"
        status = f"{action_label}（{provider}）。工作編號：{job_id}；輸出位置：{output_dir}"
        return (
            transcript_markdown(transcript),
            summary_markdown(summary, slide_insights),
            [str(path) for path in output_paths],
            status,
            job_id,
        )
    except Exception as exc:
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {"job_id": job_id}
        manifest.update(
            status="failed",
            failed_at=datetime.now().isoformat(timespec="seconds"),
            error=str(exc),
        )
        _write_json(manifest_path, manifest)
        if isinstance(exc, PipelineError) and exc.job_id:
            raise
        raise PipelineError(str(exc), job_id=job_id) from exc
