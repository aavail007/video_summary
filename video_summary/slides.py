from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .media import MediaError, ffmpeg_executable


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


@dataclass(frozen=True)
class DetectedSlide:
    index: int
    timestamp: float
    path: Path


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _frame_difference(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return float(delta.mean() / 255.0)


def select_slide_timestamps(
    frames: Iterable[tuple[float, np.ndarray]],
    *,
    change_threshold: float = 0.035,
    stability_threshold: float = 0.012,
    stable_samples: int = 2,
    minimum_interval: float = 1.5,
    max_slides: int = 250,
) -> list[float]:
    """Select stable, visually distinct pages without writing scan frames to disk."""
    previous: np.ndarray | None = None
    confirmed_frames: list[np.ndarray] = []
    stable_count = 0
    last_timestamp = -minimum_interval
    selected: list[float] = []

    for timestamp, frame in frames:
        if previous is None:
            previous = frame.copy()
            continue

        if _frame_difference(frame, previous) <= stability_threshold:
            stable_count += 1
        else:
            stable_count = 0
        previous = frame.copy()

        if len(selected) >= max_slides:
            continue
        if stable_count < stable_samples:
            continue
        if timestamp - last_timestamp < minimum_interval:
            continue
        if any(
            _frame_difference(frame, confirmed) < change_threshold
            for confirmed in confirmed_frames
        ):
            continue

        selected.append(timestamp)
        confirmed_frames.append(frame.copy())
        last_timestamp = timestamp
        stable_count = 0

    return selected


def _iter_scan_frames(
    source: Path,
    *,
    sample_fps: float,
    width: int = 320,
    height: int = 180,
) -> Iterator[tuple[float, np.ndarray]]:
    video_filter = (
        f"fps={sample_fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "format=gray"
    )
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        video_filter,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise MediaError("無法讀取 FFmpeg 的畫面輸出。")

    frame_size = width * height
    frame_index = 0
    try:
        while True:
            data = process.stdout.read(frame_size)
            if not data:
                break
            if len(data) != frame_size:
                raise MediaError("FFmpeg 回傳了不完整的影片畫面。")
            frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width)).copy()
            yield frame_index / sample_fps, frame
            frame_index += 1
    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if process.stderr:
        process.stderr.close()
    if return_code != 0:
        raise MediaError(stderr.strip() or "FFmpeg 無法掃描影片畫面。")


def _extract_full_resolution_frame(source: Path, timestamp: float, target: Path) -> None:
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(target),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0 or not target.is_file():
        raise MediaError(process.stderr.strip() or "無法擷取投影片畫面。")


def extract_unique_slides(
    source: Path,
    slides_dir: Path,
    *,
    sample_fps: float = 2.0,
    progress_callback=None,
) -> list[DetectedSlide]:
    """Keep only confirmed pages; sampled detection frames stay in memory."""
    if not is_video_file(source):
        return []

    if progress_callback:
        progress_callback("在記憶體中偵測投影片換頁")
    timestamps = select_slide_timestamps(
        _iter_scan_frames(source, sample_fps=sample_fps)
    )
    if not timestamps:
        return []

    slides_dir.mkdir(parents=True, exist_ok=True)
    slides: list[DetectedSlide] = []
    for index, timestamp in enumerate(timestamps, start=1):
        if progress_callback:
            progress_callback(f"保留確認投影片 {index}/{len(timestamps)}")
        minutes, seconds = divmod(round(timestamp), 60)
        hours, minutes = divmod(minutes, 60)
        target = slides_dir / (
            f"slide_{index:03d}_{hours:02d}-{minutes:02d}-{seconds:02d}.jpg"
        )
        _extract_full_resolution_frame(source, timestamp, target)
        slides.append(DetectedSlide(index=index, timestamp=timestamp, path=target))
    return slides
