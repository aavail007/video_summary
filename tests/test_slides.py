from pathlib import Path

import numpy as np

from video_summary import slides


def solid_frame(value: int) -> np.ndarray:
    return np.full((18, 32), value, dtype=np.uint8)


def test_selects_stable_unique_pages_and_removes_returned_duplicate() -> None:
    frames = [
        (0.0, solid_frame(10)),
        (0.5, solid_frame(10)),
        (1.0, solid_frame(10)),
        (1.5, solid_frame(10)),
        (2.0, solid_frame(10)),
        (2.5, solid_frame(220)),
        (3.0, solid_frame(220)),
        (3.5, solid_frame(220)),
        (4.0, solid_frame(220)),
        (4.5, solid_frame(220)),
        (5.0, solid_frame(10)),
        (5.5, solid_frame(10)),
        (6.0, solid_frame(10)),
        (6.5, solid_frame(10)),
        (7.0, solid_frame(10)),
    ]

    timestamps = slides.select_slide_timestamps(frames)

    assert timestamps == [2.0, 4.5]


def test_extraction_writes_only_confirmed_full_resolution_pages(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake video")
    scan_frames = [
        (0.0, solid_frame(10)),
        (0.5, solid_frame(10)),
        (1.0, solid_frame(10)),
        (1.5, solid_frame(10)),
        (2.0, solid_frame(10)),
        (2.5, solid_frame(220)),
        (3.0, solid_frame(220)),
        (3.5, solid_frame(220)),
        (4.0, solid_frame(220)),
        (4.5, solid_frame(220)),
    ]
    monkeypatch.setattr(
        slides,
        "_iter_scan_frames",
        lambda source, sample_fps: iter(scan_frames),
    )
    monkeypatch.setattr(slides, "has_video_stream", lambda source: True)

    def fake_extract(source: Path, timestamp: float, target: Path) -> None:
        target.write_bytes(f"slide at {timestamp}".encode())

    monkeypatch.setattr(slides, "_extract_full_resolution_frame", fake_extract)
    output_dir = tmp_path / "output" / "slides"

    results = slides.extract_unique_slides(source, output_dir)

    assert len(results) == 2
    assert len(list(output_dir.glob("*.jpg"))) == 2
    assert not list(tmp_path.rglob("frame_*.jpg"))


def test_extraction_skips_audio_only_container(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "youtube-audio.webm"
    source.write_bytes(b"fake audio")
    status_messages: list[str] = []

    monkeypatch.setattr(slides, "has_video_stream", lambda source: False)
    monkeypatch.setattr(
        slides,
        "_iter_scan_frames",
        lambda source, sample_fps: (_ for _ in ()).throw(
            AssertionError("audio-only source must not be scanned for video frames")
        ),
    )

    results = slides.extract_unique_slides(
        source,
        tmp_path / "output" / "slides",
        progress_callback=status_messages.append,
    )

    assert results == []
    assert status_messages == ["來源只有音訊，略過投影片擷取並繼續處理"]


def test_imports_existing_images_and_restores_filename_timestamp(
    tmp_path: Path,
) -> None:
    first = tmp_path / "slide_001_01-02-03.jpg"
    second = tmp_path / "custom.png"
    first.write_bytes(b"jpeg")
    second.write_bytes(b"png")

    results = slides.import_existing_slides(
        [str(second), str(first), str(first)],
        tmp_path / "output" / "slides",
    )

    assert len(results) == 2
    assert results[0].timestamp == 0.0
    assert results[1].timestamp == 3723
    assert results[0].path.suffix == ".png"
    assert results[1].path.suffix == ".jpg"
    assert all(result.path.is_file() for result in results)
