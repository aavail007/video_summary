from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg


_DURATION_RE = re.compile(
    r"Duration:\s*(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d+)?)"
)


class MediaError(RuntimeError):
    pass


def ffmpeg_executable() -> str:
    configured = os.getenv("FFMPEG_BINARY", "").strip()
    if configured:
        return configured
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def media_duration(path: Path) -> float:
    command = [ffmpeg_executable(), "-hide_banner", "-i", str(path)]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    diagnostic = f"{process.stdout}\n{process.stderr}"
    match = _DURATION_RE.search(diagnostic)
    if not match:
        return 0.0
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + float(match.group("seconds"))
    )


def has_video_stream(path: Path) -> bool:
    """Return whether FFmpeg can decode at least one frame from a video stream."""
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "null",
        os.devnull,
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return process.returncode == 0


def split_audio(
    source: Path,
    chunks_dir: Path,
    chunk_seconds: int = 900,
) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunks_dir / "chunk_%03d.mp3"
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        message = process.stderr.strip() or "FFmpeg 無法處理此檔案。"
        raise MediaError(message)

    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise MediaError("找不到可轉錄的音軌；檔案可能沒有聲音。")
    oversized = [path.name for path in chunks if path.stat().st_size >= 25 * 1024 * 1024]
    if oversized:
        raise MediaError(f"分段後仍超過 25 MB：{', '.join(oversized)}")
    return chunks
