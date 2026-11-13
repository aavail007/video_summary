from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .gemini_provider import transcribe_chunks_gemini
from .gemini_retry import GeminiRequestError
from .media import split_audio
from .transcription import transcribe_chunks


LIVE_ASSETS_DIR = Path(__file__).resolve().parent / "live_assets"
MAX_LIVE_CHUNK_BYTES = 20 * 1024 * 1024
LANGUAGE_LABELS = {
    "auto": "自動偵測",
    "zh": "繁體中文",
    "en": "英文",
    "ja": "日文",
}
CONTENT_TYPE_SUFFIXES = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
}


class LiveTranscriptionError(RuntimeError):
    pass


def _content_suffix(content_type: str) -> str:
    base_type = content_type.split(";", 1)[0].strip().lower()
    return CONTENT_TYPE_SUFFIXES.get(base_type, ".webm")


def transcribe_live_chunk(
    audio_data: bytes,
    *,
    content_type: str,
    provider: str,
    api_key: str,
    language_label: str,
    gemini_model: str,
    openai_model: str,
    settings: Settings,
) -> dict[str, object]:
    if not audio_data:
        raise LiveTranscriptionError("沒有收到可轉錄的音訊。")
    if len(audio_data) > MAX_LIVE_CHUNK_BYTES:
        raise LiveTranscriptionError("單次即時音訊超過 20 MB，請縮短錄音分段。")

    provider_key = provider.strip().lower()
    if provider_key not in {"gemini", "openai"}:
        raise LiveTranscriptionError("不支援的 AI 服務。")

    configured_key = (
        settings.gemini_api_key
        if provider_key == "gemini"
        else settings.openai_api_key
    )
    active_key = api_key.strip() or configured_key
    if not active_key:
        key_name = "GEMINI_API_KEY" if provider_key == "gemini" else "OPENAI_API_KEY"
        raise LiveTranscriptionError(f"請輸入 API Key，或在 .env 設定 {key_name}。")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="live-",
        dir=settings.data_dir,
    ) as temp_name:
        temp_dir = Path(temp_name)
        source = temp_dir / f"recording{_content_suffix(content_type)}"
        source.write_bytes(audio_data)
        chunks = split_audio(source, temp_dir / "chunks", chunk_seconds=300)

        if provider_key == "gemini":
            transcript = transcribe_chunks_gemini(
                chunks,
                api_key=active_key,
                model=gemini_model.strip() or settings.gemini_model,
                language_label=language_label,
                glossary="",
                source_name="即時收音",
            )
        else:
            transcript = transcribe_chunks(
                chunks,
                api_key=active_key,
                model=openai_model.strip() or settings.transcription_model,
                language_label=language_label,
                glossary="",
                source_name="即時收音",
            )

    return {
        "text": transcript.text.strip(),
        "language": transcript.language,
        "model": transcript.model,
        "segments": [segment.model_dump() for segment in transcript.segments],
    }


def register_live_routes(app, settings: Settings) -> None:
    router = APIRouter()

    @router.get("/live", include_in_schema=False)
    async def live_page():
        return FileResponse(LIVE_ASSETS_DIR / "live.html")

    @router.get("/api/live/config")
    async def live_config():
        return {
            "gemini_model": settings.gemini_model,
            "openai_model": settings.transcription_model,
            "gemini_key_configured": bool(settings.gemini_api_key),
            "openai_key_configured": bool(settings.openai_api_key),
        }

    @router.post("/api/live/transcribe")
    async def live_transcribe(request: Request):
        audio_data = await request.body()
        language_code = request.headers.get("x-language", "auto").lower()
        language_label = LANGUAGE_LABELS.get(language_code, "自動偵測")
        try:
            return await asyncio.to_thread(
                transcribe_live_chunk,
                audio_data,
                content_type=request.headers.get("content-type", "audio/webm"),
                provider=request.headers.get("x-ai-provider", "gemini"),
                api_key=unquote(request.headers.get("x-api-key", "")),
                language_label=language_label,
                gemini_model=unquote(request.headers.get("x-gemini-model", "")),
                openai_model=unquote(request.headers.get("x-openai-model", "")),
                settings=settings,
            )
        except LiveTranscriptionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GeminiRequestError as exc:
            # The server already spent the configured retry budget. The browser
            # must retain the audio for manual retry rather than multiplying it.
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": str(exc), "retryable": False},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"轉錄失敗：{exc}") from exc

    app.include_router(router)
    app.mount(
        "/live-assets",
        StaticFiles(directory=LIVE_ASSETS_DIR),
        name="live-assets",
    )
