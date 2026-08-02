import json
from pathlib import Path
from types import SimpleNamespace

from video_summary import vision
from video_summary.slides import DetectedSlide


class FakeInteractions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "slides": [
                        {
                            "index": 1,
                            "content_type": "presentation_slide",
                            "title": "架構圖",
                            "visible_text": ["輸入", "輸出"],
                            "visual_summary": "資料由輸入流向輸出。",
                        }
                    ]
                }
            )
        )


def test_gemini_slide_analysis_uses_inline_image_and_known_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "slide.jpg"
    image.write_bytes(b"jpeg")
    interactions = FakeInteractions()
    fake_client = SimpleNamespace(interactions=interactions)
    monkeypatch.setattr(
        vision.genai,
        "Client",
        lambda **kwargs: fake_client,
    )

    result = vision.analyze_slides(
        [DetectedSlide(index=1, timestamp=12.5, path=image)],
        provider="Gemini",
        api_key="test-key",
        gemini_model="gemini-test",
        openai_model="openai-test",
        reasoning_effort="low",
    )

    assert result[0].timestamp == 12.5
    assert result[0].image_file == "slides/slide.jpg"
    assert result[0].title == "架構圖"
    sent_image = interactions.calls[0]["input"][2]
    assert sent_image["type"] == "image"
    assert sent_image["mime_type"] == "image/jpeg"


def test_played_video_frames_are_removed(tmp_path: Path, monkeypatch) -> None:
    slide_image = tmp_path / "slide.jpg"
    video_image = tmp_path / "video.jpg"
    slide_image.write_bytes(b"slide")
    video_image.write_bytes(b"video")

    class MixedInteractions:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "slides": [
                            {
                                "index": 1,
                                "content_type": "presentation_slide",
                                "title": "PPT",
                                "visible_text": [],
                                "visual_summary": "簡報頁",
                            },
                            {
                                "index": 2,
                                "content_type": "played_video",
                                "title": "",
                                "visible_text": [],
                                "visual_summary": "講者播放的影片",
                            },
                        ]
                    }
                )
            )

    monkeypatch.setattr(
        vision.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(interactions=MixedInteractions()),
    )

    result = vision.analyze_slides(
        [
            DetectedSlide(index=1, timestamp=1, path=slide_image),
            DetectedSlide(index=2, timestamp=2, path=video_image),
        ],
        provider="Gemini",
        api_key="test-key",
        gemini_model="gemini-test",
        openai_model="openai-test",
        reasoning_effort="low",
    )

    assert [item.index for item in result] == [1]
    assert slide_image.is_file()
    assert not video_image.exists()
