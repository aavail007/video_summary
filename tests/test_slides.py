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
        (1.5, solid_frame(220)),
        (2.0, solid_frame(220)),
        (2.5, solid_frame(220)),
        (4.0, solid_frame(10)),
        (4.5, solid_frame(10)),
        (5.0, solid_frame(10)),
    ]

    timestamps = slides.select_slide_timestamps(frames)

    assert timestamps == [1.0, 2.5]


def test_extraction_writes_only_confirmed_full_resolution_pages(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake video")
    scan_frames = [
        (0.0, solid_frame(10)),
        (0.5, solid_frame(10)),
        (1.0, solid_frame(10)),
        (1.5, solid_frame(220)),
        (2.0, solid_frame(220)),
        (2.5, solid_frame(220)),
    ]
    monkeypatch.setattr(
        slides,
        "_iter_scan_frames",
        lambda source, sample_fps: iter(scan_frames),
    )

    def fake_extract(source: Path, timestamp: float, target: Path) -> None:
        target.write_bytes(f"slide at {timestamp}".encode())

    monkeypatch.setattr(slides, "_extract_full_resolution_frame", fake_extract)
    output_dir = tmp_path / "output" / "slides"

    results = slides.extract_unique_slides(source, output_dir)

    assert len(results) == 2
    assert len(list(output_dir.glob("*.jpg"))) == 2
    assert not list(tmp_path.rglob("frame_*.jpg"))
