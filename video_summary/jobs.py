from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Settings
from .models import TranscriptResult


JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
RESTORABLE_OPTIONS = {
    "provider", "transcription_model", "summary_model", "gemini_model",
    "reasoning_effort", "language_label", "glossary", "summary_style",
}


def resolve_job_directory(settings: Settings, reference: str) -> Path:
    """Accept a job ID or a direct path to a job under this application's jobs root."""
    reference = (reference or "").strip().strip('"').strip("'")
    jobs_root = settings.jobs_dir.resolve()
    if JOB_ID_RE.fullmatch(reference):
        candidate = jobs_root / reference
    else:
        # Both Windows and POSIX separators are accepted for relative references.
        candidate = Path(reference.replace("\\", "/"))
        if not candidate.is_absolute():
            candidate = settings.project_root / candidate
    candidate = candidate.resolve()
    if candidate.parent != jobs_root or not JOB_ID_RE.fullmatch(candidate.name):
        raise ValueError("請輸入工作編號，或目前專案 data/jobs 內的工作資料夾路徑。")
    if not candidate.is_dir() or not (candidate / "job.json").is_file():
        raise ValueError("找不到指定工作或 job.json，請確認工作資料夾仍存在。")
    return candidate


def read_job_info(settings: Settings, reference: str) -> dict:
    job_dir = resolve_job_directory(settings, reference)
    manifest = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("工作狀態檔格式不正確。")
    chunks = sorted((job_dir / "chunks").glob("chunk_*.mp3"))
    completed = 0
    inferred_settings = {}
    for index, _ in enumerate(chunks, start=1):
        cache_path = job_dir / "cache" / "transcript_chunks" / f"chunk_{index:03d}.json"
        if not cache_path.is_file():
            continue
        try:
            transcript = TranscriptResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        completed += 1
        if not inferred_settings:
            model = transcript.model
            inferred_settings = {"provider": "Gemini" if model.startswith("gemini") else "OpenAI"}
            inferred_settings["gemini_model" if model.startswith("gemini") else "transcription_model"] = model
            language = {"zh": "繁體中文", "en": "英文", "ja": "日文"}.get(transcript.language)
            if language:
                inferred_settings["language_label"] = language
    transcript_ready = (job_dir / "cache" / "transcript.json").is_file()
    stored_options = manifest.get("options")
    options = {
        name: value for name, value in stored_options.items()
        if name in RESTORABLE_OPTIONS and isinstance(value, str)
    } if isinstance(stored_options, dict) else {}
    options = {**inferred_settings, **options}
    source_path = Path(manifest.get("source_path") or job_dir / "source" / "missing")
    return {
        "job_id": job_dir.name,
        "source_name": str(manifest.get("source_name") or job_dir.name),
        "status": manifest.get("status", "interrupted"),
        "completed_chunks": completed,
        "total_chunks": len(chunks),
        "transcript_ready": transcript_ready,
        "summary_ready": (job_dir / "cache" / "summary.json").is_file(),
        "source_exists": source_path.is_file(),
        "options": options,
    }


def incomplete_job_choices(settings: Settings) -> list[tuple[str, str]]:
    """Read only: older jobs without a status are treated as interrupted."""
    choices = []
    if not settings.jobs_dir.is_dir():
        return choices
    for directory in sorted(settings.jobs_dir.iterdir(), key=lambda item: item.name, reverse=True):
        if not directory.is_dir() or not JOB_ID_RE.fullmatch(directory.name):
            continue
        try:
            info = read_job_info(settings, directory.name)
        except (ValueError, OSError):
            continue
        if info["status"] == "completed":
            continue
        progress = "逐字稿已完成" if info["transcript_ready"] else f"轉錄 {info['completed_chunks']}/{info['total_chunks']} 段"
        label = f"{info['job_id']}｜{info['source_name']}｜{progress}"
        choices.append((label, info["job_id"]))
    return choices
