import math
import struct
import wave
from pathlib import Path

from video_summary.media import media_duration, split_audio


def _write_tone(path: Path, duration_seconds: float = 2.0) -> None:
    sample_rate = 16_000
    frame_count = round(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = round(5000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        audio.writeframes(frames)


def test_ffmpeg_can_split_local_audio(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_tone(source)

    chunks = split_audio(source, tmp_path / "chunks", chunk_seconds=1)

    assert len(chunks) >= 2
    assert all(path.stat().st_size > 0 for path in chunks)
    assert 0.5 <= media_duration(chunks[0]) <= 1.5
